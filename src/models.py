<<<<<<< Updated upstream
# Định nghĩa kiến trúc mô hình
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForTokenClassification
from torchcrf import CRF
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


# =====================================================================
# PhoBERT NER Model (Fine-tuning toàn bộ, có tùy chọn CRF)
# =====================================================================
class PhoBERTNER(nn.Module):
    """
    PhoBERT-base cho NER với token classification head.
    Hỗ trợ tùy chọn CRF layer để ràng buộc output sequence.
    """
    def __init__(self, model_name, num_labels, use_crf=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        # Load pretrained PhoBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            if self.use_crf:
                # Thay -100 bằng 0 cho CRF (CRF không hỗ trợ ignore_index)
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                mask = attention_mask.bool()
                loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

    def decode(self, logits, attention_mask=None):
        if self.use_crf:
            mask = attention_mask.bool() if attention_mask is not None else None
            return self.crf.decode(logits, mask=mask)
        else:
            return torch.argmax(logits, dim=2).tolist()


# =====================================================================
# PhoBERT + LoRA (PEFT) cho NER — Tối ưu cho cấu hình thấp
# =====================================================================
class PhoBERTLoRANER(nn.Module):
    """
    PhoBERT-base với LoRA (Low-Rank Adaptation) cho fine-tuning hiệu quả.
    Chỉ huấn luyện ~0.5% tham số so với full fine-tuning.
    """
    def __init__(self, model_name, num_labels, lora_r=8, lora_alpha=16, 
                 lora_dropout=0.1, target_modules=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.num_labels = num_labels
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Cấu hình LoRA
        if target_modules is None:
            target_modules = ["query", "value"]
        
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Áp dụng LoRA
        self.bert = get_peft_model(self.base_model, lora_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
    
    def decode(self, logits, attention_mask=None):
        return torch.argmax(logits, dim=2).tolist()
    
    def print_trainable_parameters(self):
        """In số lượng tham số trainable (LoRA layers + classifier)"""
        self.bert.print_trainable_parameters()


# =====================================================================
# Bi-LSTM + CRF Model
# =====================================================================
class LSTMCRF(nn.Module):
    """
    Bi-LSTM với CRF layer cho NER.
    Kiến trúc cổ điển nhưng hiệu quả, đặc biệt với tài nguyên hạn chế.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, 
                 dropout=0.5, use_crf=True):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embedding(input_ids)
        
        # Pack padded sequences cho LSTM hiệu quả
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            if self.use_crf:
                mask = attention_mask.bool()
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
            return loss, emissions
        return emissions

    def decode(self, emissions, attention_mask=None):
        if self.use_crf:
            mask = attention_mask.bool() if attention_mask is not None else None
            return self.crf.decode(emissions, mask=mask)
        else:
            return torch.argmax(emissions, dim=2).tolist()


# =====================================================================
# Knowledge Distillation: Student Model (LSTM nhỏ)
# =====================================================================
class StudentLSTM(nn.Module):
    """
    Student model nhỏ cho Knowledge Distillation.
    Được huấn luyện bằng soft labels từ Teacher (PhoBERT).
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=1,           # Chỉ 1 layer
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
    
    def decode(self, logits, attention_mask=None):
        return torch.argmax(logits, dim=2).tolist()
=======
import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import PRETRAINED_MODEL_NAME, NUM_LABELS

class PhoBERTNER(nn.Module):
    """
    Standard PhoBERT Named Entity Recognition Model.
    Uses pre-trained vinai/phobert-base combined with a customized Linear Classifier.
    """
    def __init__(self, model_name=PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS):
        super(PhoBERTNER, self).__init__()
        self.num_labels = num_labels
        
        # Load pre-trained PhoBERT backbone
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Classifier Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Linear Classifier Head
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass for Token Classification.
        
        Args:
            input_ids (torch.Tensor): Tokenized sequence indices [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length]
            labels (torch.Tensor, optional): Ground-truth NER tag indices [batch_size, seq_length]
            
        Returns:
            dict: Dictionary containing:
                - 'logits' (torch.Tensor): Raw classification logits [batch_size, seq_length, num_labels]
                - 'loss' (torch.Tensor, optional): Token classification loss
        """
        # Get representations from PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Shape: [batch_size, seq_length, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Linear projection to get logits
        logits = self.classifier(sequence_output)  # Shape: [batch_size, seq_length, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Calculate loss only for non-padding active tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, 
                    labels.view(-1), 
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        return {"loss": loss, "logits": logits}

class PhoBERTLoRANER(nn.Module):
    """
    Memory-efficient PhoBERT Named Entity Recognition Model.
    Applies Low-Rank Adaptation (LoRA) via Hugging Face PEFT, freezing base model parameters
    and only training low-rank adapters and the linear classification head.
    """
    def __init__(
        self, 
        model_name=PRETRAINED_MODEL_NAME, 
        num_labels=NUM_LABELS, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    ):
        super(PhoBERTLoRANER, self).__init__()
        
        # 1. Initialize base model (PhoBERT + Linear Classifier Head)
        self.base_model = PhoBERTNER(model_name=model_name, num_labels=num_labels)
        
        # 2. Configure and apply LoRA wrapper on the PhoBERT backbone
        try:
            from peft import LoraConfig, get_peft_model
            
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],  # Focus on key attention matrices
                lora_dropout=lora_dropout,
                bias="none"
            )
            
            # Wrap standard phobert with LoRA, freezing its weights
            self.base_model.phobert = get_peft_model(self.base_model.phobert, peft_config)
            print("Successfully initialized PhoBERT LoRA configuration.")
            
        except ImportError:
            print("[Warning] 'peft' library is not installed. Running PhoBERTLoRANER "
                  "in full fine-tuning fallback mode without parameter freezing.")
            
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        """
        return self.base_model(input_ids, attention_mask, labels)
        
    def print_trainable_parameters(self):
        """
        Helper method to output statistics of trainable vs frozen parameters.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(
            f"Trainable params: {trainable_params:,} | "
            f"All params: {all_param:,} | "
            f"Trainable (%): {100 * trainable_params / all_param:.4f}%"
        )
>>>>>>> Stashed changes
