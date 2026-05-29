<<<<<<< Updated upstream
import os

# --- Path Configurations ---
DATA_DIR = "DATA"
TRAIN_FILE = os.path.join(DATA_DIR, "train_word.conll")
DEV_FILE = os.path.join(DATA_DIR, "dev_word.conll")
TEST_FILE = os.path.join(DATA_DIR, "test_word.conll")

CHECKPOINT_DIR = "results/checkpoints"
LOG_DIR = "results/logs"
PLOT_DIR = "results/plots"

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Label Definitions ---
# Exactly matching unique tags found in the PhoNER_COVID19 dataset
LABEL_LIST = [
    "O",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-SYMPTOM_AND_DISEASE", "I-SYMPTOM_AND_DISEASE",
    "B-LOCATION", "I-LOCATION",
    "B-DATE", "I-DATE",
    "B-PATIENT_ID", "I-PATIENT_ID",
    "B-AGE", "I-AGE",
    "B-NAME", "I-NAME",
    "B-JOB", "I-JOB",
    "B-TRANSPORTATION", "I-TRANSPORTATION",
    "B-GENDER"
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# --- Global & Transformer Hyperparameters ---
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
VAL_BATCH_SIZE = 32

EPOCHS = 10
LEARNING_RATE = 2e-5
PATIENCE = 3
WEIGHT_DECAY = 0.01

# --- LSTM & Bi-LSTM Hyperparameters ---
LSTM_EMBEDDING_DIM = 300
LSTM_HIDDEN_DIM = 256
LSTM_DROPOUT = 0.5
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS = 30

# --- Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configurations ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# --- Knowledge Distillation (KD) Configurations ---
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.5

# --- Quantization Configurations ---
QUANTIZE_BITS = 8
=======
import torch

class Config:
    def __init__(self):
        self.model_name = "vinai/phobert-base"
        self.max_len = 256
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.epochs = 5
        self.use_crf = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(args):
    return Config()
>>>>>>> Stashed changes
