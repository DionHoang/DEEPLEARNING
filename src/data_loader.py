    # Xử lý dữ liệu, tạo Dataset/DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from collections import Counter
from src import config

def read_conll_file(data_file):
    sentences, labels = [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split()
                token, tag = parts[0], parts[-1]
                tokens.append(token)
                tags.append(tag)
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

# ---------------------- PhoBERT Dataset ----------------------
class PhoBERTNERDataset(Dataset):
    def __init__(self, data_file, tokenizer, label2id, max_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.sentences = []
        self.labels = []
        self._read_data(data_file)

    def _read_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            tokens, tags = [], []
            for line in f:
                line = line.strip()
                if line == "":
                    if tokens:
                        self.sentences.append(tokens)
                        self.labels.append(tags)
                        tokens, tags = [], []
                else:
                    parts = line.split()
                    # Giả sử cột đầu là token, cột cuối là nhãn
                    token, tag = parts[0], parts[-1]
                    tokens.append(token)
                    tags.append(tag)
            if tokens:
                self.sentences.append(tokens)
                self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]

        input_ids = [self.tokenizer.cls_token_id]
        label_ids = [-100]
        
        for word, label in zip(tokens, labels):
            sub_tokens = self.tokenizer.tokenize(word)
            sub_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            if len(sub_ids) == 0:
                continue
                
            input_ids.extend(sub_ids)
            label_ids.append(self.label2id[label])
            label_ids.extend([-100] * (len(sub_ids) - 1))
            
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)
        
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
            input_ids[-1] = self.tokenizer.sep_token_id
            label_ids[-1] = -100
            
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            label_ids.extend([-100] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# ---------------------- LSTM+CRF Dataset ----------------------
class LSTMNERDataset(Dataset):
    def __init__(self, data_file, word2idx, label2id, max_len):
        self.word2idx = word2idx
        self.label2id = label2id
        self.max_len = max_len
        self.sentences = []
        self.labels = []
        self._read_data(data_file)

    def _read_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            tokens, tags = [], []
            for line in f:
                line = line.strip()
                if line == "":
                    if tokens:
                        self.sentences.append(tokens)
                        self.labels.append(tags)
                        tokens, tags = [], []
                else:
                    parts = line.split()
                    token, tag = parts[0], parts[-1]
                    tokens.append(token)
                    tags.append(tag)
            if tokens:
                self.sentences.append(tokens)
                self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx][:self.max_len]
        labels = self.labels[idx][:self.max_len]
        input_ids = [self.word2idx.get(t, self.word2idx.get('<UNK>', 1)) for t in tokens]
        label_ids = [self.label2id[l] for l in labels]
        # Padding
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.word2idx.get('<PAD>', 0)] * pad_len
        label_ids += [0] * pad_len  # 0 là 'O' (giả sử)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def build_vocab_from_files(files, max_size=50000):
    counter = Counter()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                token = line.split()[0]
                counter[token] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

def create_phobert_loaders(train_file, dev_file, test_file, model_name, label2id, max_len, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PhoBERTNERDataset(train_file, tokenizer, label2id, max_len)
    dev_dataset = PhoBERTNERDataset(dev_file, tokenizer, label2id, max_len)
    test_dataset = PhoBERTNERDataset(test_file, tokenizer, label2id, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader, tokenizer

def create_lstm_loaders(train_file, dev_file, test_file, label2id, max_len, batch_size, max_vocab_size):
    word2idx = build_vocab_from_files([train_file, dev_file], max_vocab_size)
    train_dataset = LSTMNERDataset(train_file, word2idx, label2id, max_len)
    dev_dataset = LSTMNERDataset(dev_file, word2idx, label2id, max_len)
    test_dataset = LSTMNERDataset(test_file, word2idx, label2id, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader, word2idx