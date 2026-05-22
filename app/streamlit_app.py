import os
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer

from src.config import (
    TAGS,
    IDX2TAG,
    TAG2IDX,
    NUM_LABELS,
    PRETRAINED_MODEL_NAME,
    MODEL_SAVE_DIR
)
from src.models import PhoBERTNER, PhoBERTLoRANER

# ==============================================================================
# Setup Environment & Tokenizer
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load Vietnamese word segmenter
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
    print("[Info] 'underthesea' segmenter loaded successfully.")
except ImportError:
    HAS_UNDERTHESEA = False
    print("[Warning] 'underthesea' is not installed. Text tokenization will fallback to simple space-splitting.")

# Load Tokenizer (uses Fast version if available for token-to-word alignment)
print(f"Loading tokenizer: {PRETRAINED_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)

# ==============================================================================
# Model Loading Helpers
# ==============================================================================
def load_phobert_model(model_choice):
    """
    Load PhoBERT or PhoBERT-LoRA model.
    Checks saved_models/ directory for trained checkpoints. Fallbacks to 
    pretrained base with a warning if no checkpoint is found.
    """
    model_path = ""
    is_lora = "LoRA" in model_choice
    
    if is_lora:
        model = PhoBERTLoRANER(model_name=PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
        model_path = os.path.join(MODEL_SAVE_DIR, "phobert_lora_best.pt")
    else:
        model = PhoBERTNER(model_name=PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
        model_path = os.path.join(MODEL_SAVE_DIR, "phobert_best.pt")
        
    model = model.to(device)
    
    status_msg = ""
    if os.path.exists(model_path):
        try:
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            # Handle if saved as full model or just state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"])
            elif isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            else:
                model = state_dict
            status_msg = f"✓ Loaded fine-tuned checkpoint from '{os.path.basename(model_path)}'."
        except Exception as e:
            status_msg = f"⚠ Found checkpoint but failed to load ({e}). Using initialized baseline."
    else:
        status_msg = (
            f"⚠ No checkpoint found at '{os.path.relpath(model_path)}'. "
            "Using default pretrained PhoBERT-base with initialized NER head (untrained predictions)."
        )
        
    model.eval()
    return model, status_msg

# ==============================================================================
# Prediction & Post-processing
# ==============================================================================
@torch.no_grad()
def perform_inference(text, model_choice):
    if not text.strip():
        return [], "Please enter some Vietnamese text.", ""
        
    # 1. Load Model
    model, load_status = load_phobert_model(model_choice)
    
    # 2. Segment Vietnamese Words
    if HAS_UNDERTHESEA:
        segmented_text = word_tokenize(text, format="text")
    else:
        segmented_text = text
        
    words = segmented_text.split()
    
    # 3. Tokenize and Align
    # Standard sequence tokenization matching word-based formatting
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # 4. Predict
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]  # Shape: [1, seq_length, num_labels]
    predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    
    # 5. Align predicted tags back to raw words
    word_ids = encoding.word_ids(batch_index=0)
    
    word_to_tag = {}
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # Standard approach: assign the predicted tag of the first subword token
        if word_idx not in word_to_tag:
            pred_tag_idx = predictions[idx]
            word_to_tag[word_idx] = IDX2TAG[pred_tag_idx]
            
    # 6. Format for Gradio HighlightedText
    # HighlightedText expects list of tuples: (text, label_or_none)
    highlighted_output = []
    
    for i, word in enumerate(words):
        tag = word_to_tag.get(i, "O")
        # Replace underscores in segmented words back to spaces for natural display
        display_word = word.replace("_", " ")
        
        if tag == "O":
            highlighted_output.append((display_word + " ", None))
        else:
            # Strip B- or I- prefixes for a cleaner and more beautiful UI highlight
            clean_tag = tag[2:]
            highlighted_output.append((display_word + " ", clean_tag))
            
    # Detailed JSON output for developers
    detailed_results = {
        "text_length_chars": len(text),
        "word_count": len(words),
        "segmented_text": segmented_text,
        "recognized_entities": [
            {"word": w.replace("_", " "), "tag": tag}
            for w, tag in zip(words, [word_to_tag.get(i, "O") for i in range(len(words))])
            if tag != "O"
        ]
    }
    
    return highlighted_output, load_status, detailed_results

# ==============================================================================
# Gradio Interface Styling & Theme
# ==============================================================================
custom_css = """
body {
    background-color: #0b0f19;
}
.gradio-container {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
}
.header-container {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2.5rem;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
}
.highlighted-text-container {
    background-color: #1e293b !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    padding: 1.5rem !important;
}
"""

color_map = {
    "PATIENT_ID": "#FF6B6B",      # Bright red
    "NAME": "#4DABF7",            # Soft blue
    "AGE": "#FFD43B",             # Amber yellow
    "GENDER": "#AE3EC9",          # Violet
    "JOB": "#15AABF",             # Cyan
    "LOCATION": "#40C057",        # Emerald green
    "ORGANIZATION": "#FF8787",    # Light red-pink
    "DATE": "#FAB005",            # Golden orange
    "SYMPTOM_AND_DISEASE": "#FD7E14", # Bright orange
    "TRANSPORTATION": "#7950F2"   # Deep purple
}

# ==============================================================================
# Build Gradio Blocks App
# ==============================================================================
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky", secondary_hue="indigo"), css=custom_css) as demo:
    
    # 1. Header
    gr.HTML(
        """
        <div class="header-container">
            <h1 class="header-title">🦠 PhoNER COVID-19 Analyzer</h1>
            <p class="header-subtitle">Hệ thống nhận diện thực thể y tế tiếng Việt chất lượng cao sử dụng kiến trúc PhoBERT & LoRA</p>
        </div>
        """
    )
    
    # 2. Main Work Area
    with gr.Row():
        # Left Side - Inputs & Selection
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cấu hình mô hình")
            model_selector = gr.Radio(
                choices=["PhoBERT-base (Standard)", "PhoBERT-base (LoRA)"],
                value="PhoBERT-base (Standard)",
                label="Chọn mô hình suy luận",
                info="Mô hình LoRA tối ưu tài nguyên bộ nhớ khi huấn luyện."
            )
            
            gr.Markdown("### 📝 Văn bản đầu vào")
            input_text = gr.Textbox(
                placeholder="Nhập câu văn bản tiếng Việt liên quan đến y tế hoặc COVID-19 tại đây...",
                lines=5,
                label="Câu tiếng Việt cần phân tích"
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑 Dọn dẹp", variant="secondary")
                submit_btn = gr.Button("⚡ Phân tích NER", variant="primary")
                
            gr.Markdown("💡 **Gợi ý các câu mẫu:**")
            gr.Examples(
                examples=[
                    ["Bệnh nhân 188 là L.T.H., 25 tuổi, giới tính nữ, đang điều trị tại Bệnh viện Bệnh Nhiệt đới Trung ương sau khi trở về từ Anh trên chuyến bay VN0054 ngày 22-4."],
                    ["Bộ Y tế chiều 31/7 ghi nhận thêm ca nhiễm nCoV là bệnh nhân 523, 67 tuổi, trú tại thành phố Đông Hà, Quảng Trị."],
                    ["Nam công nhân 32 tuổi có các triệu chứng ho, sốt cao và khó thở từ ngày 10-4 sau khi tiếp xúc với ca nhiễm Covid-19 tại quán bar Buddha ở Quận 2, TP.HCM."]
                ],
                inputs=[input_text]
            )

        # Right Side - Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Kết quả Nhận diện Thực thể")
            
            # Model loading status indicator
            status_output = gr.Label(
                label="Trạng thái Tải Mô hình",
                show_label=True
            )
            
            # Colored highlighted output
            highlighted_output = gr.HighlightedText(
                label="Thực thể nhận diện (Color-Coded)",
                color_map=color_map,
                elem_classes=["highlighted-text-container"]
            )
            
            # Detailed collapsible JSON view
            with gr.Accordion("🛠 Dữ liệu chi tiết (Developer JSON)", open=False):
                json_output = gr.JSON(label="Phân tích Tokens & Tags")
                
    # 3. Footer / Explanation
    with gr.Row():
        gr.Markdown(
            """
            ---
            ### ℹ️ Các loại thực thể hỗ trợ
            Hệ thống hỗ trợ tự động nhận diện và phân loại **10 nhóm thực thể** theo chuẩn PhoNER-COVID19:
            - 👤 **PATIENT_ID**: Mã số bệnh nhân (e.g. *Bệnh nhân 188*)
            - 🏷 **NAME**: Tên riêng (e.g. *L.T.H.*)
            - 📅 **AGE**: Tuổi tác (e.g. *25 tuổi*)
            - ⚥ **GENDER**: Giới tính (e.g. *nữ*)
            - 💼 **JOB**: Nghề nghiệp (e.g. *phi công*, *công nhân*)
            - 📍 **LOCATION**: Địa điểm, bệnh viện, phòng khám (e.g. *TP.HCM*, *Quảng Trị*, *Bệnh viện Bệnh Nhiệt đới*)
            - 🏢 **ORGANIZATION**: Cơ quan, tổ chức (e.g. *Bộ Y tế*, *Viện Vệ sinh dịch tễ*)
            - 📆 **DATE**: Thời gian, ngày tháng (e.g. *22-4*, *chiều 31/7*)
            - 🤒 **SYMPTOM_AND_DISEASE**: Triệu chứng và bệnh lý (e.g. *sốt cao*, *ho*, *khó thở*, *Covid-19*)
            - ✈️ **TRANSPORTATION**: Phương tiện di chuyển (e.g. *chuyến bay VN0054*)
            """
        )

    # Wire logic
    submit_btn.click(
        fn=perform_inference,
        inputs=[input_text, model_selector],
        outputs=[highlighted_output, status_output, json_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", [], "", None),
        inputs=[],
        outputs=[input_text, highlighted_output, status_output, json_output]
    )

# ==============================================================================
# Run Application
# ==============================================================================
if __name__ == "__main__":
    # Launch Gradio server
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
