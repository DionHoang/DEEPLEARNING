# Cấu hình các siêu tham số, đường dẫn
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn dữ liệu (PhoNER_COVID19 format)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_FILE = os.path.join(DATA_RAW_DIR, "train_word.conll")
DEV_FILE = os.path.join(DATA_RAW_DIR, "dev_word.conll")
TEST_FILE = os.path.join(DATA_RAW_DIR, "test_word.conll")

# Thư mục lưu trữ
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CHECKPOINTS_DIR = os.path.join(SAVED_MODELS_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Tham số chung
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 3e-5
EARLY_STOP_PATIENCE = 3
SEED = 42

# Thiết bị
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Danh sách nhãn PhoNER_COVID19
LABEL_LIST = [
    'O',
    'B-PATIENT_ID', 'I-PATIENT_ID',
    'B-NAME', 'I-NAME',
    'B-AGE', 'I-AGE',
    'B-GENDER',
    'B-JOB', 'I-JOB',
    'B-LOCATION', 'I-LOCATION',
    'B-ORGANIZATION', 'I-ORGANIZATION',
    'B-DATE', 'I-DATE',
    'B-SYMPTOM_AND_DISEASE', 'I-SYMPTOM_AND_DISEASE',
    'B-TRANSPORTATION', 'I-TRANSPORTATION',
]
NUM_LABELS = len(LABEL_LIST)
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# PhoBERT model name
PHOBERT_MODEL_NAME = "vinai/phobert-base"

# Tham số cho LSTM+CRF
LSTM_EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM = 256
LSTM_DROPOUT = 0.5
LSTM_USE_CRF = True
LSTM_MAX_VOCAB_SIZE = 50000

# ============ LoRA Configuration ============
LORA_R = 8                    # Rank của ma trận LoRA
LORA_ALPHA = 16               # Scaling factor
LORA_DROPOUT = 0.1            # Dropout cho LoRA layers
LORA_TARGET_MODULES = ["query", "value"]  # Áp dụng LoRA lên attention layers
LORA_LEARNING_RATE = 5e-4     # LR cao hơn cho LoRA vì ít tham số hơn

# ============ Knowledge Distillation Configuration ============
KD_TEMPERATURE = 3.0          # Nhiệt độ cho soft labels
KD_ALPHA = 0.5                # Trọng số giữa hard loss và soft loss
KD_STUDENT_HIDDEN_DIM = 128   # Hidden dim nhỏ hơn cho student model
