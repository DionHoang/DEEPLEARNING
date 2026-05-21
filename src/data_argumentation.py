# Data augmentation utilities for PhoNER_COVID19
import random
from typing import List, Tuple

# Vietnamese tokenizer (optional, fallback to simple split)
try:
    from pyvi import ViTokenizer
    _tokenize = lambda text: ViTokenizer.tokenize(text).split()
except Exception:
    _tokenize = lambda text: text.split()


def synonym_replacement(sentence: str, synonym_dict: dict, n: int = 2) -> str:
    """Replace *n* non‑entity tokens with a random synonym.
    *synonym_dict* should map a word to a list of its synonyms.
    The function works on tokenised words and keeps the original order.
    """
    words = _tokenize(sentence)
    # Identify candidate positions (skip words not in dict)
    candidates = [i for i, w in enumerate(words) if w in synonym_dict and synonym_dict[w]]
    if not candidates:
        return sentence
    n = min(n, len(candidates))
    chosen = random.sample(candidates, n)
    for idx in chosen:
        synonym = random.choice(synonym_dict[words[idx]])
        words[idx] = synonym
    return " ".join(words)


def back_translation(text: str, src: str = "vi", inter: str = "en") -> str:
    """Translate *text* to *inter* language and back to *src*.
    Requires ``transformers`` and the Helsinki‑NLP models.
    If the pipelines cannot be loaded, the original text is returned.
    """
    try:
        from transformers import pipeline
        translator = pipeline(
            "translation_{src}_to_{inter}".format(src=src, inter=inter),
            model=f"Helsinki-NLP/opus-mt-{src}-{inter}",
            tokenizer=f"Helsinki-NLP/opus-mt-{src}-{inter}",
            device=0 if torch.cuda.is_available() else -1,
        )
        back_translator = pipeline(
            "translation_{inter}_to_{src}".format(src=src, inter=inter),
            model=f"Helsinki-NLP/opus-mt-{inter}-{src}",
            tokenizer=f"Helsinki-NLP/opus-mt-{inter}-{src}",
            device=0 if torch.cuda.is_available() else -1,
        )
        inter_text = translator(text)[0]["translation_text"]
        back_text = back_translator(inter_text)[0]["translation_text"]
        return back_text
    except Exception:
        return text


def random_swap(words: List[str], n: int = 2) -> List[str]:
    """Swap *n* random pairs of words in *words* (in‑place)."""
    if len(words) < 2:
        return words
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def augment_sentence(sentence: str, label_seq: List[str], method: str = "swap") -> Tuple[str, List[str]]:
    """Augment *sentence* according to *method* while keeping *label_seq* aligned.

    Supported methods:
    - ``swap``: random_swap on token list (labels are swapped in the same positions).
    - ``synonym``: synonym_replacement (labels unchanged).
    - ``back_translation``: back_translation (labels unchanged).
    """
    words = _tokenize(sentence)
    if method == "swap":
        aug_words = random_swap(words.copy(), n=1)
        # When swapping words we must apply the same permutation to the label list
        # Determine the indices that were swapped.
        # Simple implementation: derive permutation by comparing original and shuffled list.
        # Here we just rebuild label list based on the new order.
        # This works because random_swap only swaps pairs.
        aug_labels = []
        for w in aug_words:
            # find the first occurrence of w in original words that hasn't been used yet
            idx = None
            for i, ow in enumerate(words):
                if ow == w and i not in aug_labels:
                    idx = i
                    break
            if idx is None:
                idx = 0
            aug_labels.append(label_seq[idx])
        return " ".join(aug_words), aug_labels
    elif method == "synonym":
        # synonym dict must be provided by the caller – we use an empty dict here
        return synonym_replacement(sentence, {}), label_seq
    elif method == "back_translation":
        return back_translation(sentence), label_seq
    else:
        raise ValueError(f"Unsupported augmentation method: {method}")


def apply_augmentation(
    input_conll: str,
    output_conll: str,
    method: str = "swap",
    augment_factor: int = 1,
) -> None:
    """Read a CoNLL‑style file, augment each sentence *augment_factor* times,
    and write the original + augmented data to *output_conll*.

    The file format is ``token <tab> label`` per line, sentences separated by an empty line.
    """
    sentences = []  # List[List[str]]
    labels = []     # List[List[str]]
    cur_words, cur_labels = [], []
    with open(input_conll, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                if cur_words:
                    sentences.append(cur_words)
                    labels.append(cur_labels)
                    cur_words, cur_labels = [], []
                continue
            parts = line.split()
            token, tag = parts[0], parts[-1]
            cur_words.append(token)
            cur_labels.append(tag)
        if cur_words:
            sentences.append(cur_words)
            labels.append(cur_labels)

    # Write original data first
    with open(output_conll, "w", encoding="utf-8") as fout:
        for sent, lab in zip(sentences, labels):
            for t, l in zip(sent, lab):
                fout.write(f"{t}\t{l}\n")
            fout.write("\n")
        # Augmented copies
        for _ in range(augment_factor):
            for sent, lab in zip(sentences, labels):
                aug_sent, aug_lab = augment_sentence(" ".join(sent), lab, method=method)
                aug_tokens = aug_sent.split()
                for t, l in zip(aug_tokens, aug_lab):
                    fout.write(f"{t}\t{l}\n")
                fout.write("\n")

# Example (commented out) ----------------------------------------------------
# apply_augmentation(
#     "data/raw/train_word.conll",
#     "data/raw/train_word_aug.conll",
#     method="swap",
#     augment_factor=2,
# )
