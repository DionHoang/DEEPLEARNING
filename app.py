import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer
from pyvi import ViTokenizer

# Gọi trực tiếp các hàm khởi tạo và cấu hình dùng chung từ pipeline hệ thống
from src import LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS, get_model_dirs, get_model

# Khởi tạo Tokenizer mặc định đồng bộ với main.py
TOKENIZER_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, keep_accents=True)

# Bộ nhớ đệm (Cache) lưu mô hình kèm thiết bị tính toán tương ứng
MODELS_CACHE = {}

# Sơ đồ ánh xạ trực quan từ UI sang các cấu hình chuẩn của hệ thống lõi
MODEL_CONFIGS = {
    "PhoBERT Gốc (Standard)": {
        "model_name": "phobert",
        "use_crf": False,
        "filename": "best_model.pt",
    },
    "PhoBERT + CRF Layer": {
        "model_name": "phobert",
        "use_crf": True,
        "filename": "best_model.pt",
    },
    "PhoBERT LoRA (Tối ưu)": {
        "model_name": "phobert-lora",
        "use_crf": False,
        "filename": "best_model.pt",
    },
    "Transformer Thuần (Scratch)": {
        "model_name": "transformer",
        "use_crf": False,
        "filename": "best_model.pt",
    },
    "Transformer + CRF Layer": {
        "model_name": "transformer",
        "use_crf": True,
        "filename": "best_model.pt",
    },
    "BiLSTM + CRF Layer": {
        "model_name": "bilstm",
        "use_crf": True,
        "filename": "best_model.pt",
    },
    "LSTM Baseline": {
        "model_name": "lstm",
        "use_crf": False,
        "filename": "best_model.pt",
    },
    "Mô hình Chưng cất (Distilled Student)": {
        "model_name": "bilstm",
        "use_crf": False,
        "filename": "best_model.pt",
    },
    "Mô hình Lượng tử hóa (Quantized INT8)": {
        "model_name": "bilstm",
        "use_crf": False,
        "filename": "bilstm_quantized_ptq.pt",
    },
}


def load_model_by_name(model_display_name):
    """
    Tự động hóa hoàn toàn việc dựng kiến trúc, tìm đường dẫn checkpoint chuẩn,
    áp dụng các bản vá thiết bị (MPS fallback) và tải trọng số tương thích.
    """
    if model_display_name in MODELS_CACHE:
        return MODELS_CACHE[model_display_name]

    cfg = MODEL_CONFIGS[model_display_name]
    model_name = cfg["model_name"]
    use_crf = cfg["use_crf"]
    filename = cfg["filename"]

    # Sử dụng hàm chuẩn của config.py để lấy chính xác thư mục checkpoints của dự án
    dirs = get_model_dirs(model_name, use_crf)
    checkpoint_path = os.path.join(dirs["checkpoints"], filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy file checkpoint tại: {checkpoint_path}"
        )

    # Cấu hình thiết bị tính toán an toàn tuân thủ bản vá lỗi số 3
    is_quantized = "quantized" in filename
    if is_quantized:
        current_device = torch.device("cpu")
    else:
        current_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Fallback an toàn cho CRF trên nền tảng Apple Silicon
        if use_crf and current_device.type == "mps":
            print(
                "[*] WARNING: Detected CRF on MPS. Forcing fallback to CPU for stability."
            )
            current_device = torch.device("cpu")

    print(
        f"[*] Khởi tạo kiến trúc model: {model_name.upper()} (CRF={use_crf}) trên thiết bị {current_device}..."
    )
    model = get_model(model_name, tokenizer.vocab_size, use_crf=use_crf)

    print(f"[*] Đang nạp trọng số từ: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=current_device)

    # Hỗ trợ cấu hình động PTQ nếu là checkpoint lượng tử hóa
    if is_quantized:
        model = torch.load(
            checkpoint_path, map_location=current_device, weights_only=False
        )
    else:
        model = get_model(model_name, tokenizer.vocab_size, use_crf=use_crf)
        ckpt = torch.load(checkpoint_path, map_location=current_device)
        model.load_state_dict(ckpt.get("model_state", ckpt))

    model.to(current_device)
    model.eval()

    # Cache lại cả instance mô hình lẫn thiết bị xử lý của nó
    MODELS_CACHE[model_display_name] = (model, current_device)
    return model, current_device


def inference_ner(model, current_device, tokenizer, text):
    """
    Tận dụng hàm `.decode()` đồng bộ có sẵn ở mọi Class mô hình trong model.py
    để xử lý đầu ra mượt mà cho cả CRF và Non-CRF.
    """
    # Khử lỗi lệch sub-word bằng cách băm từ chuẩn dịch tễ PyVi trước
    segmented_text = ViTokenizer.tokenize(text)

    inputs = tokenizer(
        segmented_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to(current_device)

    with torch.no_grad():
        # Gọi phương thức giải mã chuẩn hóa tích hợp sẵn trong model.py của bạn
        preds = model.decode(input_ids)

        # Đưa tensor hoặc list batch về dạng danh sách nhãn đơn phẳng của câu đầu vào
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    if isinstance(tokens, str):
        tokens = [tokens]

    return tokens, preds


def format_to_gradio_highlight(tokens, preds):
    """
    Lọc bỏ các token điều hướng đặc biệt, gom cụm BPE và hoàn trả văn bản sạch
    đã được gỡ bỏ ký tự gạch dưới '_' của PyVi để đưa lên Gradio UI.
    """
    highlighted_output = []
    current_word = ""
    current_label = "O"

    for token, pred_id in zip(tokens, preds):
        if token in ["<s>", "</s>", "<pad>"]:
            continue

        label = ID2LABEL.get(pred_id, "O")

        if token.endswith("@@"):
            clean_token = token[:-2]
            current_word += clean_token
            if current_label == "O" and label != "O":
                current_label = label
        else:
            current_word += token
            if current_label == "O" and label != "O":
                current_label = label

            display_label = current_label
            if display_label.startswith("B-") or display_label.startswith("I-"):
                display_label = display_label[2:]

            # Gỡ bỏ dấu gạch nối từ ghép của PyVi giúp hiển thị tự nhiên
            clean_word = current_word.replace("_", " ") + " "
            highlighted_output.append(
                (clean_word, display_label if display_label != "O" else None)
            )

            current_word = ""
            current_label = "O"

    return highlighted_output


def single_model_predict(text, model_name):
    if not text.strip():
        return [], "0.00 ms"

    start_time = time.time()
    try:
        model, current_device = load_model_by_name(model_name)
        tokens, preds = inference_ner(model, current_device, tokenizer, text)
        formatted_output = format_to_gradio_highlight(tokens, preds)
    except Exception as e:
        formatted_output = [(f"Lỗi suy luận hệ thống: {str(e)}", "ERROR")]

    latency = (time.time() - start_time) * 1000
    return formatted_output, f"{latency:.2f} ms"


def compare_models_predict(text, model_name_1, model_name_2):
    if not text.strip():
        return [], "0.00 ms", [], "0.00 ms"

    # Xử lý mô hình thứ nhất
    t1_start = time.time()
    try:
        model1, dev1 = load_model_by_name(model_name_1)
        tokens1, preds1 = inference_ner(model1, dev1, tokenizer, text)
        out1 = format_to_gradio_highlight(tokens1, preds1)
    except Exception as e:
        out1 = [(f"Lỗi mô hình 1: {str(e)}", "ERROR")]
    latency1 = (time.time() - t1_start) * 1000

    # Xử lý mô hình thứ hai
    t2_start = time.time()
    try:
        model2, dev2 = load_model_by_name(model_name_2)
        tokens2, preds2 = inference_ner(model2, dev2, tokenizer, text)
        out2 = format_to_gradio_highlight(tokens2, preds2)
    except Exception as e:
        out2 = [(f"Lỗi mô hình 2: {str(e)}", "ERROR")]
    latency2 = (time.time() - t2_start) * 1000

    return out1, f"{latency1:.2f} ms", out2, f"{latency2:.2f} ms"


# ==================== ĐỊNH NGHĨA GRADIO UI BLOCKS ====================

EXAMPLES = [
    [
        "Bệnh nhân số 1432 (BN1432) là nam giới, 25 tuổi, quốc tịch Việt Nam, có lịch trình di chuyển từ Thành phố Hồ Chí Minh đến tỉnh Vũng Tàu bằng xe khách vào ngày 25/12/2020."
    ],
    [
        "Sốt cao, ho khan, mất vị giác kèm khó thở nhẹ là các triệu chứng đặc trưng được ghi nhận tại Bệnh viện Nhiệt đới Trung ương."
    ],
    [
        "Ủy ban Nhân dân thành phố Hà Nội yêu cầu người dân tại quận Hoàn Kiếm thực hiện nghiêm túc thông điệp 5K của Bộ Y tế."
    ],
]

MODEL_CHOICES = list(MODEL_CONFIGS.keys())

with gr.Blocks(title="Vietnamese Medical NER System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩺 Hệ Thống Phân Tích Thực Thể Y Tế & Dịch Tễ Đa Kiến Trúc (PhoNER COVID-19)
        Ứng dụng hỗ trợ trích xuất thông tin tự động, cho phép đối sánh hiệu năng song song giữa các biến thể Baseline, LoRA, Custom Transformer, và các phiên bản nén (Distilled/Quantized).
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔎 Thử Nghiệm Mô Hình Đơn"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="Nhập văn bản lâm sàng / dịch tễ",
                        placeholder="Nhập câu tiếng Việt thô tại đây...",
                        lines=5,
                    )
                    model_selector = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="PhoBERT Gốc (Standard)",
                        label="Lựa chọn cấu hình và mô hình",
                    )
                    submit_btn = gr.Button("🚀 Phân Tích Thực Thể", variant="primary")

                with gr.Column(scale=1):
                    output_highlight = gr.HighlightedText(
                        label="Mã màu thực thể y tế tìm thấy", combine_adjacent=True
                    )
                    output_latency = gr.Label(label="Độ trễ suy luận")

            gr.Examples(examples=EXAMPLES, inputs=[input_text])
            submit_btn.click(
                fn=single_model_predict,
                inputs=[input_text, model_selector],
                outputs=[output_highlight, output_latency],
            )

        with gr.TabItem("📊 Bảng Đối Sánh Hiệu Năng & Độ Trễ"):
            gr.Markdown(
                "### Suy luận song song để trực quan hóa sự đánh đổi giữa Độ chính xác (Accuracy) và Tốc độ xử lý (Latency)."
            )
            compare_text = gr.Textbox(
                label="Văn bản kiểm thử đối sánh", value=EXAMPLES[0][0], lines=3
            )

            with gr.Row():
                with gr.Column():
                    comp_model_1 = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="PhoBERT Gốc (Standard)",
                        label="Mô hình Baseline (Gốc)",
                    )
                with gr.Column():
                    comp_model_2 = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="Transformer Thuần (Scratch)",
                        label="Mô hình Đối chứng",
                    )

            compare_btn = gr.Button(
                "⚡ Khởi Chạy Phân Tích Đối Sánh", variant="secondary"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### **KẾT QUẢ MÔ HÌNH 1**")
                    out_hl_1 = gr.HighlightedText(label="Thực thể tìm thấy (Mô hình 1)")
                    out_time_1 = gr.Label(label="Thời gian xử lý")
                with gr.Column():
                    gr.Markdown("#### **KẾT QUẢ MÔ HÌNH 2**")
                    out_hl_2 = gr.HighlightedText(label="Thực thể tìm thấy (Mô hình 2)")
                    out_time_2 = gr.Label(label="Thời gian xử lý")

            compare_btn.click(
                fn=compare_models_predict,
                inputs=[compare_text, comp_model_1, comp_model_2],
                outputs=[out_hl_1, out_time_1, out_hl_2, out_time_2],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
