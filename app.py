import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer
from pyvi import ViTokenizer

# Import shared initialization and configuration helpers from the core pipeline
from src import LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS, get_model_dirs, get_model

# Initialize default tokenizer (keeps consistency with main.py)
TOKENIZER_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, keep_accents=True)

# Cache storing model instances together with their compute device
MODELS_CACHE = {}

# Mapping from UI labels to internal model configuration parameters
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
    Automate model construction, locate the correct checkpoint path,
    apply device fallbacks (e.g. MPS -> CPU) and load compatible weights.
    """
    if model_display_name in MODELS_CACHE:
        return MODELS_CACHE[model_display_name]

    cfg = MODEL_CONFIGS[model_display_name]
    model_name = cfg["model_name"]
    use_crf = cfg["use_crf"]
    filename = cfg["filename"]

    # Use the config utility to find the project's checkpoint directory
    dirs = get_model_dirs(model_name, use_crf)
    checkpoint_path = os.path.join(dirs["checkpoints"], filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy file checkpoint tại: {checkpoint_path}"
        )

    # Determine compute device with safe fallbacks
    is_quantized = "quantized" in filename
    if is_quantized:
        current_device = torch.device("cpu")
    else:
        current_device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Safe fallback for CRF on Apple Silicon (MPS)
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

    # Support dynamic PTQ checkpoint handling when file indicates quantization
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

    # Cache the model instance along with its compute device
    MODELS_CACHE[model_display_name] = (model, current_device)
    return model, current_device


def inference_ner(model, current_device, tokenizer, text):
    """
    Use the model's built-in `.decode()` to obtain predictions consistently
    for both CRF and non-CRF models.
    """
    # Reduce sub-word splitting issues by tokenizing using PyVi first
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
        # Call the model's integrated decode method
        preds = model.decode(input_ids)

        # Convert batch tensor/list to a flat list of labels for the input sentence
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    if isinstance(tokens, str):
        tokens = [tokens]

    return tokens, preds


def format_to_gradio_highlight(tokens, preds):
    """
    Remove special navigation tokens, merge BPE pieces, and return cleaned
    words (with PyVi underscores removed) for display in the Gradio UI.
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

            # Remove PyVi's underscore word joins for natural display
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


# ==================== GRADIO UI BLOCKS DEFINITION ====================

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
    gr.Markdown("""
        # 🩺 Hệ Thống Phân Tích Thực Thể Y Tế & Dịch Tễ Đa Kiến Trúc (PhoNER COVID-19)
        Ứng dụng hỗ trợ trích xuất thông tin tự động, cho phép đối sánh hiệu năng song song giữa các biến thể Baseline, LoRA, Custom Transformer, và các phiên bản nén (Distilled/Quantized).
        """)

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
