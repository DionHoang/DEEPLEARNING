import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def read_conll(file_path):
    """
    Parse CoNLL/TXT formatted files for Named Entity Recognition.

    Sentences are separated by blank lines. Each non-empty line contains a word
    and its corresponding tag separated by space or tab.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.

    Returns
    ---
    list of dict
        A list where each item is a dictionary with:
        - "words": list of str (tokens in a sentence)
        - "tags": list of str (NER labels aligned with words)
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL file not found: {file_path}")

    sentences = []
    words, tags = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append({"words": words, "tags": tags})
                    words, tags = [], []
                continue

            parts = line.split()
            if len(parts) >= 2:
                words.append(parts[0])
                tags.append(parts[-1])

    if words:
        sentences.append({"words": words, "tags": tags})

    return sentences


class VietnameseNERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition (NER) using HuggingFace tokenizers.

    This dataset performs:
    - Word-level input encoding using HF tokenizer
    - Subword tokenization handling via is_split_into_words=True
    - Label alignment using word_ids()
    - Padding and truncation to fixed max length
    - Masking subword tokens with ignore_index (-100)

    Parameters
    ---
    sentences : list of dict
        Output from read_conll(), each item contains:
        - "words": list of tokens
        - "tags": list of NER labels

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer (e.g. PhoBERT, BERT, XLM-R).

    max_len : int
        Maximum sequence length for padding/truncation.

    label2id : dict
        Mapping from label string to integer ID.

    Attributes
    ---
    ignore_index : int
        Label value used to ignore subword tokens in loss computation.
    """

    def __init__(self, sentences, tokenizer, max_len, label2id):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.ignore_index = -100

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns a single training sample.
        """
        sentence = self.sentences[idx]
        words = sentence["words"]
        tags = sentence["tags"]

        # Bắt đầu với token [CLS]
        input_ids = [self.tokenizer.cls_token_id]
        labels = [self.ignore_index]

        for word, tag in zip(words, tags):
            # Tokenize cắt từ nguyên bản thành các sub-words (BPE)
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue

            token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            input_ids.extend(token_ids)

            # Gán label gốc cho sub-word đầu tiên. Các sub-word sau gán ignore_index (-100)
            labels.append(self.label2id.get(tag, self.label2id["O"]))
            labels.extend([self.ignore_index] * (len(token_ids) - 1))

        # Kết thúc với token [SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        labels.append(self.ignore_index)

        # Xử lý Truncation (Cắt bớt nếu vượt quá max_len)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len - 1] + [self.tokenizer.sep_token_id]
            labels = labels[: self.max_len - 1] + [self.ignore_index]

        # Xử lý Padding (Bổ sung nếu ngắn hơn max_len)
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            labels.extend([self.ignore_index] * padding_length)

        # Convert sang Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input_ids, labels


def get_dataloader(file_path, tokenizer, batch_size, max_len, label2id, shuffle=False):
    """
    Helper function to load a CoNLL file, create a VietnameseNERDataset,
    and return a PyTorch DataLoader.

    This function acts as a pipeline wrapper:
    CoNLL file → parsed sentences → Dataset → DataLoader.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer used for subword encoding.

    batch_size : int
        Number of samples per batch.

    max_len : int
        Maximum sequence length for padding/truncation.

    label2id : dict
        Mapping from label string to integer IDs.

    shuffle : bool
        Whether to shuffle the dataset each epoch.

    Returns
    ---
    DataLoader
        PyTorch DataLoader object ready for training or evaluation.
    """

    sentences = read_conll(file_path)
    dataset = VietnameseNERDataset(sentences, tokenizer, max_len, label2id)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )

    return dataloader
