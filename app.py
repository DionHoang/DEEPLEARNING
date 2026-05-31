import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer

# Import các cấu hình và lớp mô hình từ module src của dự án
from src import (
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    MAX_SEQ_LENGTH,
    PhoBERTModel,
    PhoBERTLORA,
    BiLSTMModel,
    LSTMModel
)

# Thiết lập thiết bị chạy (Ưu tiên GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo Tokenizer mặc định của PhoBERT
# Lưu ý: vinai/phobert-base yêu cầu dữ liệu đã được tách từ (word-segmented)
TOKENIZER_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Bộ nhớ đệm (Cache) để tránh việc load lại mô hình nhiều lần gây chậm hệ thống
MODELS_CACHE = {}

def load_model_by_name(model_name):
    """
    Hàm tải mô hình từ thư mục checkpoints và lưu vào bộ nhớ cache.
    """
    if model_name in MODELS_CACHE:
        return MODELS_CACHE[model_name]
    
    print(f"[*] Đang tải mô hình {model_name} lên {device}...")
    
    # Định nghĩa đường dẫn checkpoint dựa vào sơ đồ thư mục dự án
    checkpoint_mapping = {
        "PhoBERT Gốc (Standard)": ("phobert_best.pt", False),
        "PhoBERT LoRA (Tối ưu)": ("phobert_lora_best.pt", False),
        "BiLSTM-CRF": ("bilstm_best.pt", True),
        "Mô hình Chưng cất (Distilled Student)": ("student_model_distilled.pt", False),
        "Mô hình Lượng tử hóa (Quantized 8-bit)": ("quantized_model.pt", False)
    }
    
    if model_name not in checkpoint_mapping:
        raise ValueError(f"Không tìm thấy cấu hình cho mô hình: {model_name}")
        
    filename, use_crf = checkpoint_mapping[model_name]
    checkpoint_path = os.path.join("checkpoints", filename)
    
    # Khởi tạo instance của class mô hình tương ứng
    if "PhoBERT Gốc" in model_name:
        model = PhoBERTModel(num_labels=NUM_LABELS, use_crf=use_crf)
    elif "LoRA" in model_name:
        model = PhoBERTLORA(num_labels=NUM_LABELS, use_crf=use_crf)
    elif "BiLSTM" in model_name or "Chưng cất" in model_name:
        # Giả định vocab_size tương thích với PhoBERT Tokenizer nếu dùng chung embedding
        model = BiLSTMModel(vocab_size=tokenizer.vocab_size, num_labels=NUM_LABELS, use_crf=use_crf)
    elif "Quantized" in model_name:
        # Mô hình quantized thường được lưu toàn bộ cấu trúc hoặc load qua hàm đặc thù
        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path, map_location="cpu")
            model.eval()
            MODELS_CACHE[model_name] = model
            return model
        else:
            raise FileNotFoundError(f"Không tìm thấy file checkpoint lượng tử hóa tại {checkpoint_path}")

    # Tải trọng số weights từ file checkpoint (.pt)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    else:
        print(f"[!] Cảnh báo: Không tìm thấy file {checkpoint_path}, mô hình sẽ sử dụng trọng số ngẫu nhiên!")

    model.to(device)
    model.eval()
    MODELS_CACHE[model_name] = model
    return model

def inference_ner(model, tokenizer, text):
    """
    Thực hiện tokenize và suy luận NER từ văn bản thô đầu vào.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        # Kiểm tra xem mô hình có dùng tầng giải mã CRF hay không
        if hasattr(model, "use_crf") and model.use_crf:
            if hasattr(model, "decode"):
                preds = model.decode(input_ids, attention_mask)
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy().tolist()[0]
                elif isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
                    preds = preds[0]
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=-1).squeeze().tolist()
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=-1).squeeze().tolist()
            
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    return tokens, preds

def format_to_gradio_highlight(tokens, preds):
    """
    Chuyển đổi danh sách tokens và predictions thành định dạng đầu ra của Gradio HighlightedText:
    [(w1, label1), (w2, label2), ...]
    """
    highlighted_output = []
    current_word = ""
    current_label = "O"
    
    for token, pred_id in zip(tokens, preds):
        # Bỏ qua các token đặc biệt hệ thống
        if token in ["<s>", "</s>", "<pad>"]:
            continue
            
        label = ID2LABEL.get(pred_id, "O")
        
        # Xử lý cơ chế ghép lại các Subword bị phân rã bởi BPE (kết thúc bằng @@) của PhoBERT
        if token.endswith("@@"):
            clean_token = token[:-2]
            current_word += clean_token
            if current_label == "O" and label != "O":
                current_label = label
        else:
            current_word += token
            if current_label == "O" and label != "O":
                current_label = label
                
            # Đã gom đủ một từ hoàn chỉnh, chuẩn hóa nhãn hiển thị trực quan (bỏ tiền tố B- / I-)
            display_label = current_label
            if display_label.startswith("B-") or display_label.startswith("I-"):
                display_label = display_label[2:]
                
            highlighted_output.append((current_word + " ", display_label if display_label != "O" else None))
            
            # Reset trạng thái cho từ tiếp theo
            current_word = ""
            current_label = "O"
            
    return highlighted_output

def single_model_predict(text, model_name):
    """Xử lý suy luận cho giao diện đơn mô hình"""
    if not text.strip():
        return [], "0.00 ms"
        
    start_time = time.time()
    try:
        model = load_model_by_name(model_name)
        tokens, preds = inference_ner(model, tokenizer, text)
        formatted_output = format_to_gradio_highlight(tokens, preds)
    except Exception as e:
        formatted_output = [(f"Đã xảy ra lỗi khi thực hiện suy luận: {str(e)}", "ERROR")]
        
    latency = (time.time() - start_time) * 1000  # Đổi sang mili-giây
    return formatted_output, f"{latency:.2f} ms"

def compare_models_predict(text, model_name_1, model_name_2):
    """Xử lý suy luận song song cho giao diện đối sánh cấu hình mô hình"""
    if not text.strip():
        return [], "0.00 ms", [], "0.00 ms"
        
    # Chạy mô hình thứ nhất
    t1_start = time.time()
    try:
        model1 = load_model_by_name(model_name_1)
        tokens1, preds1 = inference_ner(model1, tokenizer, text)
        out1 = format_to_gradio_highlight(tokens1, preds1)
    except Exception as e:
        out1 = [(f"Lỗi hệ thống khi chạy mô hình 1: {str(e)}", "ERROR")]
    latency1 = (time.time() - t1_start) * 1000

    # Chạy mô hình thứ hai
    t2_start = time.time()
    try:
        model2 = load_model_by_name(model_name_2)
        tokens2, preds2 = inference_ner(model2, tokenizer, text)
        out2 = format_to_gradio_highlight(tokens2, preds2)
    except Exception as e:
        out2 = [(f"Lỗi hệ thống khi chạy mô hình 2: {str(e)}", "ERROR")]
    latency2 = (time.time() - t2_start) * 1000

    return out1, f"{latency1:.2f} ms", out2, f"{latency2:.2f} ms"

# ==================== THIẾT KẾ GIAO DIỆN GRADIO UI ====================

# Danh sách văn bản mẫu chuẩn dịch tễ PhoNER COVID-19 để test nhanh
EXAMPLES = [
    ["Bệnh nhân số 1432 (BN1432) là nam giới, 25 tuổi, quốc tịch Việt Nam, có lịch trình di chuyển từ Thành phố Hồ Chí Minh đến tỉnh Vũng Tàu bằng xe khách vào ngày 25/12/2020."],
    ["Sốt cao, ho khan, mất vị giác kèm khó thở nhẹ là các triệu chứng đặc trưng được ghi nhận tại Bệnh viện Nhiệt đới Trung ương."],
    ["Ủy ban Nhân dân thành phố Hà Nội yêu cầu người dân tại quận Hoàn Kiếm thực hiện nghiêm túc thông điệp 5K của Bộ Y tế."]
]

MODEL_CHOICES = [
    "PhoBERT Gốc (Standard)", 
    "PhoBERT LoRA (Tối ưu)", 
    "BiLSTM-CRF", 
    "Mô hình Chưng cất (Distilled Student)",
    "Mô hình Lượng tử hóa (Quantized 8-bit)"
]

with gr.Blocks(title="Vietnamese Medical NER Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩺 Giao Diện Nhận Diện Thực Thể Tên Riêng Y Tế & Dịch Tễ (Vietnamese NER)
        Ứng dụng hỗ trợ trích xuất thông tin tự động từ văn bản tiếng Việt thuộc miền dữ liệu dịch tễ **PhoNER COVID-19**.
        """
    )
    
    with gr.Tabs():
        # TAB 1: TRẢI NGHIỆM ĐƠN MÔ HÌNH
        with gr.TabItem("🔎 Demo Đơn Mô Hình"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="Nhập văn bản cần trích xuất thực thể", 
                        placeholder="Nhập hoặc chọn câu mẫu bên dưới...", 
                        lines=5
                    )
                    model_selector = gr.Dropdown(
                        choices=MODEL_CHOICES, 
                        value="PhoBERT Gốc (Standard)", 
                        label="Lựa chọn kiến trúc mô hình"
                    )
                    submit_btn = gr.Button("🚀 Thực Hiện Nhận Diện", variant="primary")
                    
                with gr.Column(scale=1):
                    output_highlight = gr.HighlightedText(
                        label="Kết quả nhận diện thực thể (Entity Highlights)",
                        combine_adjacent=True
                    )
                    output_latency = gr.Label(label="Thời gian xử lý (Inference Latency)")
            
            # Gắn danh sách mẫu dưới form nhập liệu
            gr.Examples(examples=EXAMPLES, inputs=[input_text])
            submit_btn.click(
                fn=single_model_predict, 
                inputs=[input_text, model_selector], 
                outputs=[output_highlight, output_latency]
            )

        # TAB 2: ĐỐI SÁNH HIỆU NĂNG SONG SONG (GỐC VS COMPRESSED)
        with gr.TabItem("📊 So Sánh Các Biến Thể Mô Hình"):
            gr.Markdown("### So sánh song song kết quả đầu ra và độ trễ phản hồi (Latency) giữa 2 kiến trúc khác nhau.")
            compare_text = gr.Textbox(
                label="Nhập văn bản kiểm thử so sánh", 
                value=EXAMPLES[0][0], 
                lines=3
            )
            
            with gr.Row():
                with gr.Column():
                    comp_model_1 = gr.Dropdown(choices=MODEL_CHOICES, value="PhoBERT Gốc (Standard)", label="Mô hình 1 (Baseline)")
                with gr.Column():
                    comp_model_2 = gr.Dropdown(choices=MODEL_CHOICES, value="Mô hình Lượng tử hóa (Quantized 8-bit)", label="Mô hình 2 (Compressed)")
            
            compare_btn = gr.Button("⚡ Bắt Đầu Phân Tích & Đối Sánh", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### **KẾT QUẢ MÔ HÌNH 1**")
                    out_hl_1 = gr.HighlightedText(label="Thực thể tìm thấy (Mô hình 1)")
                    out_time_1 = gr.Label(label="Độ trễ Mô hình 1")
                with gr.Column():
                    gr.Markdown("#### **KẾT QUẢ MÔ HÌNH 2**")
                    out_hl_2 = gr.HighlightedText(label="Thực thể tìm thấy (Mô hình 2)")
                    out_time_2 = gr.Label(label="Độ trễ Mô hình 2")
                    
            compare_btn.click(
                fn=compare_models_predict,
                inputs=[compare_text, comp_model_1, comp_model_2],
                outputs=[out_hl_1, out_time_1, out_hl_2, out_time_2]
            )

# Khởi chạy ứng dụng Web trên local port 7860
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)