import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .config import (
    LoRAConfig,
    NUM_LABELS,
    LABEL2ID,
    LSTMConfig,
    TransformerConfig,
    BERTConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from .utils import setup_logger
import math

logger = setup_logger("model")


class NERLoss(nn.Module):
    """
    Unified loss function that routes labels and logits depending on whether
    the model utilizes a CRF decoding layer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, labels, inputs=None):
        if hasattr(self.model, "use_crf") and self.model.use_crf:
            return self.model.crf_loss(outputs, labels, inputs)
        else:
            return self.ce(outputs.view(-1, outputs.shape[-1]), labels.view(-1))


def get_model(model_name, vocab_size, pad_token_id=1, use_crf=False):
    """Instantiate the requested model class."""
    if model_name == "lstm":
        lstm_cfg = LSTMConfig()
        logger.info(f"Instantiating LSTM model (CRF={use_crf})")
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    elif model_name == "bilstm":
        logger.info(f"Instantiating Bidirectional LSTM model (CRF={use_crf})")
        lstm_cfg = LSTMConfig()
        return BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    elif model_name == "phobert":
        logger.info(f"Instantiating PhoBERT model (CRF={use_crf})")
        return PhoBERTModel(
            num_labels=NUM_LABELS, pad_token_id=pad_token_id, use_crf=use_crf
        )
    elif model_name == "phobert-lora":
        logger.info(f"Instantiating PhoBERT + LoRA model (CRF={use_crf})")
        return PhoBERTLoRA(
            num_labels=NUM_LABELS, pad_token_id=pad_token_id, use_crf=use_crf
        )
    elif model_name == "transformer":
        logger.info(f"Instantiating Scratch Transformer model (CRF={use_crf})")
        tf_cfg = TransformerConfig()
        return TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=tf_cfg.embedding_dim,
            nhead=tf_cfg.nhead,
            num_layers=tf_cfg.num_layers,
            dim_feedforward=tf_cfg.dim_feedforward,
            num_labels=NUM_LABELS,
            max_seq_length=tf_cfg.max_seq_length,
            dropout=tf_cfg.dropout,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class CRFLayer(nn.Module):
    """
    A native, fully vectorized PyTorch implementation of a Linear-Chain Conditional Random Field (CRF) layer.
    This avoids any dynamic C++ compilation issues and works out of the box on CPU/GPU.
    """

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # Transition parameters: transitions[i, j] is the score of transitioning from tag j to tag i.
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None):
        """
        Compute the negative log-likelihood of the gold tag sequence.

        Parameters
        ---
        emissions : torch.Tensor
            Logits of shape (batch_size, seq_len, num_tags).
        tags : torch.Tensor
            Gold tags of shape (batch_size, seq_len).
        mask : torch.Tensor
            Boolean mask of shape (batch_size, seq_len) where 1 indicates real token and 0 indicates padding/ignore.

        Returns
        ---
        torch.Tensor
            Scalar loss.
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        log_partition = self._compute_log_partition(emissions, mask)
        gold_score = self._compute_gold_score(emissions, tags, mask)

        # Return negative log-likelihood loss
        return torch.mean(log_partition - gold_score)

    def decode(self, emissions, mask=None):
        """
        Find the highest-scoring (Viterbi) tag sequence.

        Parameters
        ---
        emissions : torch.Tensor
            Logits of shape (batch_size, seq_len, num_tags).
        mask : torch.Tensor
            Boolean mask of shape (batch_size, seq_len).

        Returns
        ---
        list of list of int
            The best decoded path for each sequence in the batch.
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        return self._viterbi_decode(emissions, mask)

    def _compute_log_partition(self, emissions, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # mask: (batch_size, seq_len)
        batch_size, seq_len, num_tags = emissions.shape

        # Transpose to (seq_len, batch_size, num_tags) for sequential iteration
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # Initialize alpha with start transitions + first emission
        alpha = (
            self.start_transitions.view(1, num_tags) + emissions[0]
        )  # (batch_size, num_tags)

        for i in range(1, seq_len):
            # Broadcast alpha (batch_size, num_tags, 1) and transitions (1, num_tags, num_tags)
            # and emissions[i] (batch_size, 1, num_tags)
            alpha_t = alpha.unsqueeze(2)  # (batch_size, num_tags, 1)
            emit_t = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)
            trans_t = self.transitions.transpose(0, 1).unsqueeze(
                0
            )  # (1, num_tags, num_tags)

            # Sum of scores: alpha_t + transitions + emit_t
            # (batch_size, num_tags, num_tags)
            scores = alpha_t + trans_t + emit_t

            # Log-sum-exp over source tags (dim 1)
            next_alpha = torch.logsumexp(scores, dim=1)  # (batch_size, num_tags)

            # Only update alpha where mask is 1
            mask_i = mask[i].unsqueeze(1)  # (batch_size, 1)
            alpha = torch.where(mask_i, next_alpha, alpha)

        # Add end transitions
        alpha = alpha + self.end_transitions.view(1, num_tags)
        return torch.logsumexp(alpha, dim=1)

    def _compute_gold_score(self, emissions, tags, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # tags: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        batch_size, seq_len, num_tags = emissions.shape

        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # Score at step 0
        score = (
            self.start_transitions[tags[0]]
            + emissions[0, torch.arange(batch_size), tags[0]]
        )

        for i in range(1, seq_len):
            # Transition score from tag at step i-1 to tag at step i
            transition_score = self.transitions[tags[i], tags[i - 1]]
            # Emission score for tag at step i
            emission_score = emissions[i, torch.arange(batch_size), tags[i]]

            # Increment score if masked
            next_score = score + transition_score + emission_score
            score = torch.where(mask[i], next_score, score)

        # Add end transition score for the last active token in each sequence
        # We need to find the index of the last active token for each sequence
        # mask is (seq_len, batch_size)
        last_indices = mask.long().sum(dim=0) - 1  # (batch_size,)
        last_tags = tags[last_indices, torch.arange(batch_size)]  # (batch_size,)

        score = score + self.end_transitions[last_tags]
        return score

    def _viterbi_decode(self, emissions, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # mask: (batch_size, seq_len)
        batch_size, seq_len, num_tags = emissions.shape

        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        # Initialize viterbi variables
        viterbi = (
            self.start_transitions.view(1, num_tags) + emissions[0]
        )  # (batch_size, num_tags)
        backpointers = []

        for i in range(1, seq_len):
            # Broadcast viterbi (batch_size, num_tags, 1) and transitions (1, num_tags, num_tags)
            viterbi_t = viterbi.unsqueeze(2)  # (batch_size, num_tags, 1)
            trans_t = self.transitions.transpose(0, 1).unsqueeze(
                0
            )  # (1, num_tags, num_tags)

            # (batch_size, num_tags, num_tags)
            scores = viterbi_t + trans_t

            # Max score and argmax over source tags (dim 1)
            max_scores, argmaxes = torch.max(scores, dim=1)  # (batch_size, num_tags)

            # Add emission scores
            next_viterbi = max_scores + emissions[i]

            # Only update where mask is 1
            mask_i = mask[i].unsqueeze(1)  # (batch_size, 1)
            viterbi = torch.where(mask_i, next_viterbi, viterbi)

            # Save backpointers (clamped/ignored for masked tokens)
            backpointers.append(argmaxes)

        # Add end transitions
        viterbi = viterbi + self.end_transitions.view(1, num_tags)

        # Trace best paths
        best_paths = []
        for b in range(batch_size):
            # Find sequence length (number of active tokens)
            seq_l = mask[:, b].long().sum().item()
            if seq_l == 0:
                best_paths.append([0])
                continue

            # Get best tag for last active token
            best_tag = torch.argmax(viterbi[b]).item()
            path = [best_tag]

            # Backtrack
            for i in range(seq_l - 2, -1, -1):
                # backpointers[i] is of shape (batch_size, num_tags)
                best_tag = backpointers[i][b, best_tag].item()
                path.append(best_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths


class LSTMModel(nn.Module):
    """
    A standard LSTM model with an optional CRF classification layer for NER baseline.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_labels,
        dropout=0.5,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.pad_token_id
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        # Create mask: ignore padding / ignore indices (-100)
        mask = labels != -100
        # For CRF, we replace -100 in labels with 0 so index is valid during score calculation,
        # but the mask ensures these positions do not contribute to score/partition.
        clean_labels = labels.clone()
        clean_labels[clean_labels == -100] = LABEL2ID.get("O", 0)
        return self.crf(logits, clean_labels, mask)

    def decode(self, input_ids):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                mask = input_ids != self.pad_token_id
                return self.crf.decode(logits, mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()


class BiLSTMModel(nn.Module):
    """
    A Bidirectional LSTM model with an optional CRF classification layer for NER.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_labels,
        dropout=0.5,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.pad_token_id
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embeds = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.clone()
        clean_labels[clean_labels == -100] = LABEL2ID.get("O", 0)
        return self.crf(logits, clean_labels, mask)

    def decode(self, input_ids):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                mask = input_ids != self.pad_token_id
                return self.crf.decode(logits, mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()


class PhoBERTModel(nn.Module):
    """
    Pretrained PhoBERT sequence tagger with an optional CRF classification layer.
    """

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=NUM_LABELS,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        # Load the base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # Automatically generate attention mask from input_ids (1 is PhoBERT's padding token)
        attention_mask = (input_ids != self.pad_token_id).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.clone()
        clean_labels[clean_labels == -100] = LABEL2ID.get("O", 0)
        return self.crf(logits, clean_labels, mask)

    def decode(self, input_ids):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                mask = input_ids != 1
                return self.crf.decode(logits, mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()

    # Override state_dict to bypass PEFT's aggressive filtering if you want
    # to save the entire model in PyTorch's native format.
    # def state_dict(self, *args, **kwargs):
    #     # Force return of all parameters, not just adapters
    #     return super().state_dict(*args, **kwargs)


class PhoBERTLoRA(nn.Module):
    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_labels=NUM_LABELS,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.base_model = PhoBERTModel(
            model_name=model_name,
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            use_crf=use_crf,
        )

        cfg = LoRAConfig()
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=["query", "key", "value", "dense"],
        )

        self.base_model.bert = get_peft_model(self.base_model.bert, peft_config)

        # Unfreeze classifier and crf
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

        if use_crf and hasattr(self.base_model, "crf"):
            for param in self.base_model.crf.parameters():
                param.requires_grad = True

    @property
    def use_crf(self):
        return self.base_model.use_crf

    def forward(self, input_ids):
        return self.base_model(input_ids)

    def crf_loss(self, logits, labels, input_ids=None):
        return self.base_model.crf_loss(logits, labels, input_ids)

    def decode(self, input_ids):
        return self.base_model.decode(input_ids)


class PositionalEncoding(nn.Module):
    """Positional encoding layer for the from-scratch Transformer."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Pure Transformer (encoder-only) model with optional CRF for NER tasks."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        num_layers,
        dim_feedforward,
        num_labels,
        max_seq_length=256,
        dropout=0.1,
        pad_token_id=1,
        use_crf=False,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_length, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        for mod in encoder_layer.modules():
            if isinstance(mod, nn.MultiheadAttention):
                mod.out_proj = nn.Linear(
                    mod.out_proj.in_features, mod.out_proj.out_features
                )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.classifier = nn.Linear(embedding_dim, num_labels)

        if self.use_crf:
            self.crf = CRFLayer(num_labels)

    def forward(self, input_ids):
        # Create key padding mask from input_ids (1 is PhoBERT's pad token)
        src_key_padding_mask = input_ids == self.pad_token_id

        embeds = self.embedding(input_ids)
        embeds = self.pos_encoder(embeds)

        try:
            # Pass mask argument positionally (None for the second mask) for safety
            encoder_out = self.transformer_encoder(embeds, None, src_key_padding_mask)
        except (AttributeError, TypeError):
            # Fallback if passing the mask causes an error (e.g., in quantized models)
            encoder_out = self.transformer_encoder(embeds)

        logits = self.classifier(encoder_out)
        return logits

    def crf_loss(self, logits, labels, input_ids=None):
        mask = labels != -100
        clean_labels = labels.clone()
        clean_labels[clean_labels == -100] = LABEL2ID.get("O", 0)
        return self.crf(logits, clean_labels, mask)

    def decode(self, input_ids):
        # Perform inference without altering training/eval mode
        with torch.no_grad():
            logits = self.forward(input_ids)
            if self.use_crf:
                mask = input_ids != self.pad_token_id
                return self.crf.decode(logits, mask)
            else:
                return torch.argmax(logits, dim=-1).cpu().tolist()
