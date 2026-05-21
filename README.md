# Vietnamese NER Project – PhoNER COVID-19

## 🎯 Giới thiệu Dự án
Kho lưu trữ này triển khai hệ thống **Nhận diện Thực thể Tiếng Việt (Named Entity Recognition - NER)**, được xây dựng dựa trên tập dữ liệu **PhoNER_COVID19**.
Dự án cung cấp hai kiến trúc mô hình chính:

| Mô hình | Mô tả | Khi nào nên dùng |
|-------|-------------|-------------|
| **PhoBERT-base** (Transformer) | Fine-tuning toàn bộ mô hình ngôn ngữ `vinai/phobert-base`. | Cần độ chính xác cao nhất, yêu cầu GPU. |
| **Bi-LSTM + CRF** | Mô hình hồi quy nhẹ kết hợp tầng giải mã CRF. | Tài nguyên hạn chế / cần tốc độ suy luận nhanh. |
| **PhoBERT-LoRA** (tùy chọn) | Áp dụng Low-Rank Adaptation (LoRA) cho PhoBERT, chỉ cập nhật ~0.5% tham số. | Môi trường hạn chế bộ nhớ (VRAM thấp). |
| **Student LSTM (KD)** | Mô hình LSTM nhỏ gọi được huấn luyện qua Knowledge Distillation (chưng cất tri thức) từ nhãn mềm của PhoBERT. | Cần mô hình cực nhẹ nhưng vẫn giữ được hiệu năng tốt. |

Dự án tuân theo cấu trúc được yêu cầu trong Đồ án và bao gồm một **ứng dụng web Streamlit** để demo khả năng suy luận trực tiếp.

---

## 📂 Cấu trúc thư mục

```
ner_project/
├── data/
│   ├── raw/                # Dữ liệu CoNLL gốc (train_word.conll, dev_word.conll, test_word.conll)
│   ├── processed/          # Dữ liệu đã tiền xử lý
│   └── external/           # Tài nguyên bổ sung (embeddings, etc.)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Khám phá dữ liệu, thống kê, xem câu mẫu
│   ├── 02_model_prototyping.ipynb   # Huấn luyện thử PhoBERT / LSTM-CRF, vẽ biểu đồ Loss & F1
│   ├── 03_error_analysis.ipynb     # Đánh giá cấp độ thực thể, ma trận nhầm lẫn, phân tích lỗi
│   └── 04_compare_models.ipynb     # Biểu đồ so sánh Precision/Recall/F1 giữa các mô hình
├── src/
│   ├── __init__.py
│   ├── config.py           # Siêu tham số, đường dẫn, cấu hình LoRA & KD
│   ├── data_loader.py      # Đọc CoNLL, class Dataset, DataLoaders
│   ├── models.py           # PhoBERT, PhoBERT-LoRA, Bi-LSTM-CRF, Student LSTM
│   ├── train.py            # Vòng lặp huấn luyện, early-stopping, lưu checkpoint
│   ├── data_argumentation.py # Tăng cường dữ liệu (Synonym, Back-translation)
│   └── utils.py            # Metrics, vẽ biểu đồ, ma trận nhầm lẫn, đánh giá entity-level
├── saved_models/
│   ├── phobert_best.pt     # Checkpoint tốt nhất của PhoBERT
│   └── lstm_best.pt        # Checkpoint tốt nhất của LSTM
├── results/
│   ├── metrics.csv         # CSV lưu loss/F1 theo từng epoch
│   ├── plots/              # Các biểu đồ kết quả
│   └── predictions.txt     # Kết quả dự đoán trên tập test
├── app/
│   └── streamlit_app.py    # Giao diện demo Streamlit
├── README.md               # Tài liệu bạn đang đọc
├── requirements.txt        # Các thư viện Python cần thiết
└── .gitignore
```

---

## 🛠️ Cài đặt & Thiết lập

1. **Tạo môi trường ảo (Khuyến nghị)**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   # Kích hoạt trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt các thư viện yêu cầu**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu PhoNER_COVID19**
   Đảm bảo các file dữ liệu (`train_word.conll`, `dev_word.conll`, `test_word.conll`) được đặt đúng trong thư mục `data/raw/`.

---

## 🚀 Hướng dẫn Chạy Dự án

### 1. Khám phá Dữ liệu và Phân tích
Mở các notebook trong thư mục `notebooks/` để xem phân tích dữ liệu, huấn luyện mẫu và so sánh mô hình:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Huấn luyện Mô hình
Sử dụng script `src/train.py` để huấn luyện. Chạy lệnh sau từ thư mục gốc của dự án:

**Huấn luyện PhoBERT:**
```bash
python -m src.train --model phobert --epochs 10 --batch_size 16 --lr 2e-5
```

**Huấn luyện LSTM-CRF:**
```bash
python -m src.train --model lstm --epochs 30 --batch_size 32 --lr 1e-3
```

**Các tham số dòng lệnh hỗ trợ:**
- `--model`: `phobert` hoặc `lstm` (bắt buộc)
- `--epochs`: Số lượng epochs
- `--batch_size`: Kích thước batch
- `--lr`: Tốc độ học (Learning Rate)
- `--patience`: Số epoch chờ cho Early Stopping

### 3. Demo Giao diện Web (Streamlit)
Sau khi huấn luyện và có file model trong `saved_models/`, bạn có thể chạy ứng dụng web để thử nghiệm:
```bash
streamlit run app/streamlit_app.py
```
Ứng dụng cho phép bạn nhập văn bản tiếng Việt, chọn mô hình (PhoBERT hoặc LSTM) và hiển thị kết quả nhận diện thực thể với màu sắc trực quan.

---

## ⚙️ Các Tinh chỉnh Có Thể Chỉnh Sửa (Hyperparameters)

Toàn bộ cấu hình hệ thống nằm trong file `src/config.py`. Dưới đây là các tham số bạn có thể điều chỉnh để tối ưu hóa mô hình:

### 🔧 Tham số chung
- `MAX_SEQ_LENGTH = 256`: Độ dài tối đa của chuỗi đầu vào. Tăng lên nếu câu dài, nhưng tốn bộ nhớ hơn.
- `BATCH_SIZE = 16`: Giảm xuống nếu bị lỗi Out Of Memory (OOM) trên GPU.
- `EPOCHS = 10` và `LEARNING_RATE = 2e-5`: Tùy chỉnh để mô hình hội tụ tốt nhất.

### 🧠 Cấu hình LoRA (Cho PhoBERT)
Nếu bạn muốn huấn luyện mô hình ngôn ngữ lớn trên GPU yếu, hãy bật LoRA bằng cách sử dụng `PhoBERTLoRANER` trong `models.py`.
- `LORA_R = 8`: Hạng (rank) của LoRA. Tăng `r` (ví dụ: 16, 32) giúp mô hình học được nhiều hơn nhưng tốn tài nguyên hơn.
- `LORA_ALPHA = 16`: Hệ số scale cho LoRA. Thường đặt gấp đôi `LORA_R`.
- `LORA_DROPOUT = 0.1`: Giúp giảm thiểu overfitting.

### 🎓 Cấu hình Knowledge Distillation
Nếu bạn muốn chưng cất tri thức từ PhoBERT sang LSTM.
- `KD_TEMPERATURE = 3.0`: Độ "mềm" của nhãn (soft labels). Nhiệt độ cao hơn làm phẳng phân bố xác suất của mô hình giáo viên.
- `KD_ALPHA = 0.5`: Trọng số cân bằng giữa loss chuẩn (hard loss) và loss chưng cất (soft loss).

### 🔄 Cấu hình LSTM
- `LSTM_EMBEDDING_DIM = 300`: Kích thước vector nhúng từ.
- `LSTM_HIDDEN_DIM = 256`: Kích thước lớp ẩn. Tăng lên giúp mô hình biểu diễn tốt hơn nhưng chậm hơn.
- `LSTM_DROPOUT = 0.5`: Chống overfitting.
- `LSTM_USE_CRF = True`: Bật/tắt lớp CRF. CRF giúp mô hình học được mối quan hệ giữa các nhãn liền kề (ví dụ: `I-PER` không bao giờ đi sau `B-LOC`).
