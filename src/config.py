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

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"

# Ensure directories exist
for path in [CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

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


# --- Global & Transformer Hyperparameters ---
@dataclass
class TransformerConfig:
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


# --- Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configurations ---
@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1


# --- Knowledge Distillation (KD) Configurations ---
@dataclass
class KDConfig:
    temperature: float = 3.0
    alpha: float = 0.5


# --- Quantization Configurations ---
@dataclass
class QuantConfig:
    bits: int = 8
