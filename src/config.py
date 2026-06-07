from pathlib import Path
from dataclasses import dataclass

# --- Base directory ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_DIR = BASE_DIR / "DATA"

TRAIN_FILE = DATA_DIR / "train_word.conll"
DEV_FILE = DATA_DIR / "dev_word.conll"
TEST_FILE = DATA_DIR / "test_word.conll"

# --- Output paths ---
OUTPUT_DIR = BASE_DIR / "results"


def get_model_dirs(model_name: str, use_crf: bool = False):
    """Create and return the subdirectory structure for a given model."""
    crf_suffix = "_crf" if use_crf else ""
    model_dir = OUTPUT_DIR / f"{model_name}{crf_suffix}"

    dirs = {
        "base": str(model_dir),
        "checkpoints": str(model_dir / "checkpoints"),
        "logs": str(model_dir / "logs"),
        "plots": str(model_dir / "plots"),
        "tensorboard": str(model_dir / "tensorboard"),
    }

    # Ensure all output directories exist
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return dirs


# --- Label Definitions ---
# Exactly matching unique tags found in the PhoNER_COVID19 dataset
LABEL_LIST = [
    "O",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-SYMPTOM_AND_DISEASE",
    "I-SYMPTOM_AND_DISEASE",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-PATIENT_ID",
    "I-PATIENT_ID",
    "B-AGE",
    "I-AGE",
    "B-NAME",
    "I-NAME",
    "B-JOB",
    "I-JOB",
    "B-TRANSPORTATION",
    "I-TRANSPORTATION",
    "B-GENDER",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# --- Native Transformer Hyperparameters ---
@dataclass
class TransformerConfig:
    embedding_dim: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_length: int = 256
    batch_size: int = 32
    val_batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 5e-4
    patience: int = 5


# --- BERT & PhoBERT Hyperparameters ---
@dataclass
class BERTConfig:
    max_seq_length: int = 256
    batch_size: int = 16
    val_batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    patience: int = 3
    weight_decay: float = 0.01


# --- LSTM & Bi-LSTM Hyperparameters ---
@dataclass
class LSTMConfig:
    embedding_dim: int = 300
    hidden_dim: int = 256
    dropout: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 30
    max_seq_length: int = 256
    batch_size: int = 32
    val_batch_size: int = 32
    patience: int = 5


# --- Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configurations ---
@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    warmup_ratio: float = 0.1


# --- Knowledge Distillation (KD) Configurations ---
@dataclass
class KDConfig:
    temperature: float = 4.0
    alpha: float = 0.7


# --- Quantization Configurations ---
@dataclass
class QuantConfig:
    bits: int = 8
