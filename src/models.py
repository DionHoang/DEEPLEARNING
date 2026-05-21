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
