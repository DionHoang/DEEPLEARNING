<<<<<<< Updated upstream
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def read_conll(file_path):
    """
    Parse CoNLL/TXT formatted files for Named Entity Recognition.
    Sentences are separated by blank lines. Each non-empty line contains a word and its tag
    separated by space or tab.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.

    Returns
    ---
    list of dict
        A list where each item is a dictionary with keys "words" (list of str)
        and "tags" (list of str).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL file not found: {file_path}")

    sentences = []
    current_words = []
    current_tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append({
                        "words": current_words,
                        "tags": current_tags
                    })
                    current_words = []
                    current_tags = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[-1]  # The tag is always the last column
                current_words.append(word)
                current_tags.append(tag)
                
        # Append the last sentence if the file doesn't end with a blank line
        if current_words:
            sentences.append({
                "words": current_words,
                "tags": current_tags
            })

    return sentences


class VietnameseNERDataset(Dataset):
    """
    A custom PyTorch Dataset for Vietnamese NER.
    It tokenizes sentences using a Hugging Face tokenizer (like PhoBERT/ViBERT),
    handles subword BPE Tokenization, and implements Label Alignment.
    """
    def __init__(self, sentences, tokenizer, max_len, label2id):
        """
        Parameters
        ---
        sentences : list of dict
            Parsed sentences from read_conll.
        tokenizer : PreTrainedTokenizer
            Hugging Face tokenizer (e.g. PhoBERT).
        max_len : int
            Maximum sequence length.
        label2id : dict
            Mapping from label name to label ID.
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.ignore_index = -100

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]["words"]
        tags = self.sentences[idx]["tags"]

        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

        input_ids = [bos_id]
        labels = [self.ignore_index]

        for word, tag in zip(words, tags):
            # Tokenize the word into subwords using standard tokenizer.tokenize
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

            tag_id = self.label2id.get(tag, self.label2id.get("O", 0))

            # The first subword gets the actual entity tag
            input_ids.extend(word_token_ids)
            labels.append(tag_id)
            # Subsequent subwords get the ignore index (-100)
            labels.extend([self.ignore_index] * (len(word_token_ids) - 1))

        # Truncate sequence if it exceeds max_len - 1 (leave 1 space for EOS token)
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[:self.max_len - 1]
            labels = labels[:self.max_len - 1]

        # Append EOS token
        input_ids.append(eos_id)
        labels.append(self.ignore_index)

        # Pad sequence to max_len
        if len(input_ids) < self.max_len:
            padding_len = self.max_len - len(input_ids)
            input_ids.extend([pad_id] * padding_len)
            labels.extend([self.ignore_index] * padding_len)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def get_dataloader(file_path, tokenizer, batch_size, max_len, label2id, shuffle=False):
    """
    Helper function to load a CoNLL file, create a VietnameseNERDataset, and return a DataLoader.

    Parameters
    ---
    file_path : str
        Path to the CoNLL file.
    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer.
    batch_size : int
        Batch size.
    max_len : int
        Maximum sequence length.
    label2id : dict
        Mapping from label name to label ID.
    shuffle : bool
        Whether to shuffle the dataset.

    Returns
    ---
    DataLoader
        PyTorch DataLoader.
    """
    sentences = read_conll(file_path)
    dataset = VietnameseNERDataset(sentences, tokenizer, max_len, label2id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Safe default to avoid multi-processing pickling issues on MacOS
    )
    return dataloader
=======
import torch
from torch.utils.data import Dataset, DataLoader
import json

def read_conll_data(file_path):
    """
    Read data in CoNLL format.
    Each line contains a word and its label separated by a space or tab.
    Sentences are separated by a blank line.
    
    Args:
        file_path (str): Path to the CoNLL file.
        
    Returns:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        labels (list of list of str): Corresponding list of labels.
    """
    sentences = []
    labels = []
    
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    # Assuming format: word label
                    # The last column is the label, the first is the word
                    word = parts[0]
                    label = parts[-1]
                    current_sentence.append(word)
                    current_labels.append(label)
                    
        # Add the last sentence if the file doesn't end with an empty line
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
            
    return sentences, labels

def extract_label_list(labels_list):
    """
    Extract the list of unique labels from the dataset,
    ensuring the label 'O' is always at the beginning (index 0).
    
    Args:
        labels_list (list of list of str): List of labels for the entire dataset.
        
    Returns:
        list of str: List of unique labels.
    """
    unique_labels = set()
    for labels in labels_list:
        for label in labels:
            unique_labels.add(label)
            
    label_list = list(unique_labels)
    # Ensure 'O' is the first label if it exists
    if 'O' in label_list:
        label_list.remove('O')
        label_list = ['O'] + sorted(label_list)
    else:
        label_list = sorted(label_list)
        
    return label_list

class NERDataset(Dataset):
    """
    PyTorch Dataset for the NER task.
    Handles tokenization and label alignment when using a BPE tokenizer (like PhoBERT/ViBERT).
    """
    def __init__(self, texts, tags, tokenizer, max_len, label2id):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        words = self.texts[idx]
        labels = self.tags[idx]
        
        # Use the tokenizer to tokenize the words (which are already pre-split)
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Handle Label Alignment
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP], [PAD]
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # New word (first subword)
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                # Subsequent subwords of the same word
                # PhoBERT/ViBERT assigns -100 to subsequent subwords
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label_ids, dtype=torch.long)
        
        return item

def create_dataloader(texts, tags, tokenizer, max_len, label2id, batch_size, shuffle=True, num_workers=0):
    """
    Create a DataLoader from texts and tags.
    
    Args:
        texts (list): List of sentences.
        tags (list): Corresponding list of labels.
        tokenizer: Pre-trained tokenizer.
        max_len (int): Maximum sequence length.
        label2id (dict): Mapping from label to id.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = NERDataset(texts, tags, tokenizer, max_len, label2id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
>>>>>>> Stashed changes
