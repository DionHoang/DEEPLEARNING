# Vietnamese NER Project - PhoNER COVID-19

## Giới thiệu Dự án

Kho lưu trữ này triển khai hệ thống Nhận diện Thực thể Tiếng Việt (Named Entity Recognition - NER) chuyên biệt cho dịch tễ học và thông tin dịch bệnh COVID-19, được xây dựng dựa trên tập dữ liệu PhoNER_COVID19.

## Tập dữ liệu PhoNER_COVID19

PhoNER_COVID19 là tập dữ liệu chuẩn phục vụ cho bài toán nhận diện thực thể tên tiếng Việt trong lĩnh vực dịch tễ học và thông tin dịch bệnh COVID-19. Tập dữ liệu bao gồm khoảng 35.000 thực thể được gán nhãn thủ công trên 10.000 câu văn bản. Có 10 loại thực thể (nhãn) được định nghĩa chi tiết dưới đây:

| Nhãn (Label) | Tên Thực Thể | Định nghĩa chi tiết |
| :--- | :--- | :--- |
| `PATIENT_ID` | Mã bệnh nhân | Mã định danh duy nhất của bệnh nhân nhiễm COVID-19 tại Việt Nam (ví dụ: bệnh nhân thứ X). |
| `NAME` | Tên người | Tên của bệnh nhân hoặc những người tiếp xúc trực tiếp/liên quan đến bệnh nhân. |
| `AGE` | Tuổi | Tuổi của bệnh nhân hoặc những người liên quan. |
| `GENDER` | Giới tính | Giới tính của bệnh nhân hoặc những người liên quan. |
| `JOB` | Nghề nghiệp | Nghề nghiệp của bệnh nhân hoặc những người liên quan. |
| `LOCATION` | Địa điểm | Các địa điểm hoặc địa chỉ mà bệnh nhân đã từng xuất hiện hoặc đi qua. |
| `ORGANIZATION` | Tổ chức | Các tổ chức liên quan đến bệnh nhân (ví dụ: công ty, cơ quan chính phủ, bệnh viện, v.v.). |
| `SYMPTOM_AND_DISEASE` | Triệu chứng & Bệnh lý | Triệu chứng bệnh nhân gặp phải hoặc các bệnh nền bệnh nhân có trước khi nhiễm bệnh. |
| `TRANSPORTATION` | Phương tiện di chuyển | Phương tiện di chuyển cụ thể mà bệnh nhân đã sử dụng (ví dụ: số hiệu chuyến bay, biển số xe, v.v.). |
| `DATE` | Ngày tháng | Bất kỳ mốc thời gian hay ngày tháng nào xuất hiện trong câu văn. |

*Lưu ý: Tập dữ liệu tuân thủ quy tắc không gán nhãn cho các thực thể lồng nhau (nested entities).*

## Kiến trúc mô hình hỗ trợ

Dự án hỗ trợ đa dạng các kiến trúc mạng từ cơ bản đến các mô hình ngôn ngữ lớn tiên tiến, kết hợp các kỹ thuật tối ưu hóa và nén mô hình:

| Mô hình | Mô tả | Vai trò |
| :--- | :--- | :--- |
| PhoBERT-base | Fine-tuning toàn bộ mô hình ngôn ngữ vinai/phobert-base | Đạt độ chính xác cao nhất (SOTA). |
| PhoBERT-LoRA | Tích hợp Low-Rank Adaptation (LoRA), chỉ huấn luyện < 1% tham số | Tiết kiệm bộ nhớ VRAM, chống overfitting. |
| Transformer (Scratch) | Kiến trúc Transformer Encoder học từ đầu (không pre-trained) | Thử nghiệm hiệu năng mạng tự xây dựng. |
| BiLSTM | Mô hình mạng hồi quy hai chiều truyền thống | Nhẹ, chạy nhanh trên CPU và GPU. |
| LSTM (Unidirectional) | Mô hình mạng hồi quy một chiều cơ bản | Làm mốc so sánh hiệu năng (Baseline) cơ bản nhất. |
| Tầng CRF (Tùy chọn) | Tầng giải mã Conditional Random Field (CRF) tự viết thuần PyTorch | Ràng buộc logic nhãn đầu ra, tăng mạnh F1. |
| Chưng cất tri thức (KD) | Truyền tri thức từ PhoBERT (Teacher) sang BiLSTM (Student) | Tạo mô hình cực nhẹ nhưng có độ chính xác cao. |
| Quantization (PTQ/QAT) | Nén mô hình sang định dạng số nguyên INT8 (Dynamic/QAT) | Giảm dung lượng mô hình 4 lần, tăng tốc suy luận. |

---

## Bảng kết quả thực nghiệm (Leaderboard)

Kết quả đánh giá chi tiết trên tập kiểm thử (Test Set) chuẩn PhoNER ở cả cấp độ Token và Thực thể (Entity BIO):

| Kiến trúc Mô hình | Cấu hình | Sử dụng CRF | Token Accuracy | Token Macro F1 | Entity BIO F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LSTM** | Float (Baseline) | Không | 0.9526 | 0.7163 | 0.8435 |
| | Float (Baseline) | Có | 0.9553 | 0.7352 | 0.8658 |
| | Distilled (KD) | Không | 0.9484 | 0.6345 | 0.8259 |
| | Distilled (KD) | Có | 0.9537 | 0.7122 | 0.8672 |
| | QAT (INT8) | Không | 0.7587 | 0.0725 | 0.0153 |
| | QAT (INT8) | Có | 0.8388 | 0.2680 | 0.3772 |
| | PTQ (INT8) | Không | 0.9527 | 0.7165 | 0.8434 |
| | PTQ (INT8) | Có | 0.9554 | 0.7357 | 0.8659 |
| **BiLSTM** | Float (Baseline) | Không | 0.9622 | 0.7752 | 0.8758 |
| | Float (Baseline) | Có | 0.9603 | 0.7518 | 0.8834 |
| | Distilled (KD) | Không | 0.9555 | 0.6820 | 0.8667 |
| | Distilled (KD) | Có | 0.9592 | 0.7363 | 0.8868 |
| | QAT (INT8) | Không | 0.8349 | 0.3155 | 0.3128 |
| | QAT (INT8) | Có | 0.8966 | 0.4515 | 0.6542 |
| | PTQ (INT8) | Không | 0.9622 | 0.7645 | 0.8758 |
| | PTQ (INT8) | Có | 0.9604 | 0.7535 | 0.8840 |
| **Transformer (Scratch)** | Float (Baseline) | Không | 0.9437 | 0.7184 | 0.7918 |
| | Float (Baseline) | Có | 0.9470 | 0.7369 | 0.8215 |
| | Distilled (KD) | Không | 0.9424 | 0.6647 | 0.7920 |
| | Distilled (KD) | Có | 0.9397 | 0.6677 | 0.7915 |
| | QAT (INT8) | Không | 0.9342 | 0.6631 | 0.7573 |
| | QAT (INT8) | Có | 0.9375 | 0.6812 | 0.7836 |
| | PTQ (INT8) | Không | 0.9436 | 0.7191 | 0.7918 |
| | PTQ (INT8) | Có | 0.9471 | 0.7373 | 0.8219 |
| **PhoBERT-base** | Float (Baseline) | Không | 0.9825 | 0.8406 | 0.9497 |
| | Float (Baseline) | Có | 0.9825 | 0.8437 | 0.9466 |
| | LoRA (r=16) | Không | 0.9819 | 0.8351 | 0.9442 |
| | LoRA (r=16) | Có | **0.9837** | **0.8981** | **0.9546** |

Nhận xét:

1. Mô hình dựa trên PhoBERT-base đạt hiệu quả vượt trội so với các kiến trúc học từ đầu (LSTM, BiLSTM, Transformer), khẳng định sức mạnh của các mô hình ngôn ngữ pre-trained lớn trên tiếng Việt.
2. Việc sử dụng tầng giải mã CRF giúp cải thiện độ chính xác và đảm bảo tính hợp lệ logic của chuỗi nhãn đầu ra, đặc biệt hữu ích với các mô hình nhỏ hơn.
3. Kỹ thuật chưng cất tri thức (Knowledge Distillation) giúp mô hình học sinh (Student - BiLSTM) duy trì hiệu năng cao tiệm cận mô hình giáo viên (Teacher - PhoBERT) nhưng giảm thiểu tối đa tài nguyên và thời gian suy luận.
4. Lượng tử hóa sau huấn luyện (PTQ) giữ nguyên hoặc chỉ giảm rất nhẹ độ chính xác trong khi giảm dung lượng checkpoint đáng kể. Ngược lại, Quantization Aware Training (QAT) đối với LSTM/BiLSTM gặp hiện tượng giảm hiệu năng mạnh do mất mát thông tin khi biểu diễn trọng số nhị phân hoặc nguyên hóa từ đầu trong tập dữ liệu nhỏ.

---

## Cấu trúc thư mục thực tế

```text
DEEPLEARNING/
├── DATA/                   # Dữ liệu CoNLL gốc (train_word.conll, dev_word.conll, test_word.conll)
├── notebook/               # Thư mục chứa các Notebook thực nghiệm trên Colab và phân tích EDA
│   ├── colab_run_full.ipynb # Notebook chạy huấn luyện, KD, QAT/PTQ toàn diện trên Google Colab
│   └── eda.ipynb           # Phân tích dữ liệu khám phá (Exploratory Data Analysis)
├── src/
│   ├── __init__.py
│   ├── config.py           # Siêu tham số tập trung (BERT, LSTM, Transformer, LoRA, KD, QAT)
│   ├── dataset.py          # Xây dựng Dataloader và quản lý từ vựng cho LSTM
│   ├── data_augmentor.py   # Làm giàu dữ liệu huấn luyện (Entity Substitution)
│   ├── model.py            # Khai báo các mô hình (PhoBERT, LoRA, BiLSTM, Scratch Transformer, CRF)
│   ├── trainer.py          # Class Trainer hướng đối tượng (BaseTrainer, DistillationTrainer, QuantizationTrainer)
│   ├── utils.py            # Các học phần phụ trợ tính toán Metrics, vẽ biểu đồ, log
│   ├── quantize_utils.py   # Các tiện ích nén mô hình tĩnh/động INT8
│   └── engine.py           # Cốt lõi điều phối chạy các chế độ
├── results/                # Nơi lưu trữ checkpoint, log, biểu đồ theo từng mô hình riêng biệt
├── .env                    # Tập tin cấu hình biến môi trường
├── main.py                 # Tệp đầu vào điều phối chính (CLI)
├── requirements.txt        # Thư viện yêu cầu
└── README.md
```

---

## Hướng dẫn thiết lập và chạy kịch bản thực nghiệm

### 1. Thiết lập môi trường ảo và cài đặt thư viện (.venv)

Trước khi thực hiện các kịch bản huấn luyện hoặc đánh giá, bạn cần thiết lập môi trường ảo để tránh xung đột thư viện:

#### Trên Windows (CMD)

```cmd
# Tạo môi trường ảo .venv
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate.bat

# Nâng cấp pip và cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Trên Linux / macOS

```bash
# Tạo môi trường ảo .venv
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Nâng cấp pip và cài đặt thư viện
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Chạy các kịch bản thực nghiệm (main.py)

Mọi câu lệnh đều được chạy thông qua file main.py ở thư mục gốc sau khi đã kích hoạt môi trường ảo.

### 1. Huấn luyện mô hình thường (Float)

Sử dụng --mode train. Tham số --model hỗ trợ: phobert, phobert-lora, transformer, bilstm, lstm (hoặc all để chạy toàn bộ).

* Huấn luyện PhoBERT + CRF:

  ```bash
  python main.py --model phobert --mode train --use_crf
  ```

* Huấn luyện BiLSTM + CRF:

  ```bash
  python main.py --model bilstm --mode train --use_crf --epochs 30 --lr 1e-3
  ```

### 2. Đánh giá mô hình (Evaluate)

Sử dụng --mode evaluate. Nếu không chỉ định --checkpoint, hệ thống sẽ tự động quét checkpoint tốt nhất trong thư mục results/<model_name>/checkpoints/.

* Đánh giá mô hình PhoBERT-LoRA + CRF:

  ```bash
  python main.py --model phobert-lora --mode evaluate --use_crf
  ```

### 3. Chưng cất tri thức (Knowledge Distillation - KD)

Chưng cất tri thức từ mô hình giáo viên PhoBERT sang học sinh BiLSTM:

```bash
python main.py --model bilstm --mode distill --use_crf --checkpoint results/phobert_crf/checkpoints/best_model.pt
```

### 4. Lượng tử hóa mô hình sau huấn luyện (Post-Training Quantization - PTQ)

Nén mô hình sang định dạng INT8 động (chỉ áp dụng cho các dòng LSTM/Transformer Scratch):

```bash
python main.py --model bilstm --mode quantize --use_crf --checkpoint results/bilstm_crf/checkpoints/best_model.pt
```

### 5. Huấn luyện nhận biết lượng tử (Quantization Aware Training - QAT)

Huấn luyện kèm mô phỏng sai số lượng tử hóa để bảo toàn tối đa độ chính xác của mô hình nén:

```bash
python main.py --model bilstm --mode train_qat --use_crf
```
