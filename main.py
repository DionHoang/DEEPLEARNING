import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from src import (
    TRAIN_FILE,
    DEV_FILE,
    TEST_FILE,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    CHECKPOINT_DIR,
    PLOT_DIR,
    TransformerConfig,
    LSTMConfig,
    LoRAConfig,
    KDConfig,
    QuantConfig,
    read_conll,
    get_dataloader,
    LSTMModel,
    BiLSTMModel,
    PhoBERTModel,
    PhoBERTLoRA,
    BaseTrainer,
    DistillationTrainer,
    QuantizationTrainer,
    setup_logger,
    compute_entity_level_metrics,
    compute_metrics,
    plot_loss_curve,
    plot_f1_curve,
    plot_confusion_matrix,
    plot_entity_distribution,
    save_metrics_csv,
    save_predictions,
    get_error_analysis,
    NERDataAugmentor,
    VietnameseNERDataset,
)

from torch.utils.data import DataLoader

import src.quantize_utils as quantize_utils

# Set up main logger
logger = setup_logger("main_orchestrator")


class NERLoss(nn.Module):
    """
    Unified loss function that routes labels and logits depending on whether
    the model utilizes a CRF decoding layer.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, labels, inputs=None):
        if hasattr(self.model, "use_crf") and self.model.use_crf:
            return self.model.crf_loss(outputs, labels, inputs)
        else:
            return self.ce(outputs.view(-1, outputs.shape[-1]), labels.view(-1))


def get_model(model_name, vocab_size, use_crf=False):
    """Instantiate the requested model class."""
    if model_name == "lstm":
        lstm_cfg = LSTMConfig()
        logger.info(f"Instantiating LSTM model (CRF={use_crf})")
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            use_crf=use_crf,
        )
    elif model_name == "bilstm":
        logger.info(f"Instantiating Bidirectional LSTM model (CRF={use_crf})")
        lstm_cfg = LSTMConfig()
        return BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=lstm_cfg.embedding_dim,
            hidden_dim=lstm_cfg.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=lstm_cfg.dropout,
            use_crf=use_crf,
        )
    elif model_name == "phobert":
        logger.info(f"Instantiating PhoBERT model (CRF={use_crf})")
        return PhoBERTModel(num_labels=NUM_LABELS, use_crf=use_crf)
    elif model_name == "phobert-lora":
        logger.info(f"Instantiating PhoBERT + LoRA model (CRF={use_crf})")
        return PhoBERTLoRA(num_labels=NUM_LABELS, use_crf=use_crf)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_evaluation(
    model, dataloader, tokenizer, id2label, dataset_sentences, save_pred_path=None
):
    """Run rigorous token-level and entity-level evaluation."""
    model.eval()
    all_preds = []
    all_labels = []

    device = next(model.parameters()).device
    logger.info("Running evaluation cycle...")

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            if hasattr(model, "use_crf") and model.use_crf:
                # CRF models return Viterbi paths as lists
                batch_preds = model.decode(inputs)

                # Align batch_preds length with labels (which are padded to max_len)
                for preds, gold_labels in zip(batch_preds, labels.cpu().tolist()):
                    active_len = sum(1 for g in gold_labels if g != -100)
                    # Extend/truncate if necessary to match active labels
                    preds = preds[:active_len] + [-100] * (
                        len(gold_labels) - len(preds[:active_len])
                    )
                    all_preds.append(preds)
                    all_labels.append(gold_labels)
            else:
                # Non-CRF standard prediction
                logits = model(inputs)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                for p, g in zip(preds, labels.cpu().tolist()):
                    all_preds.append(p)
                    all_labels.append(g)

    # Flatten for token-level metrics
    flat_preds = []
    flat_labels = []
    for p_seq, l_seq in zip(all_preds, all_labels):
        for p, l in zip(p_seq, l_seq):
            flat_preds.append(p)
            flat_labels.append(l)

    # Compute Token-Level Classification Report
    token_report = compute_metrics(flat_preds, flat_labels, LABEL_LIST)

    # Compute Entity-Level precision, recall, and F1 (BIO span-based evaluation)
    entity_report = compute_entity_level_metrics(all_preds, all_labels, id2label)

    # Save predictions to file if requested
    if save_pred_path and dataset_sentences:
        tokens_list = [sent["words"] for sent in dataset_sentences]
        # Align prediction tags with tokens (excluding paddings)
        clean_trues = []
        clean_preds = []
        for p_seq, l_seq in zip(all_preds, all_labels):
            t_seq_clean = [l for l in l_seq if l != -100]
            # Lọc trực tiếp prediction tương ứng với label hợp lệ
            p_seq_clean = [p for p, l in zip(p_seq, l_seq) if l != -100]

            clean_trues.append(t_seq_clean)
            clean_preds.append(p_seq_clean)

        save_predictions(
            tokens_list, clean_trues, clean_preds, save_pred_path, id2label
        )

    return token_report, entity_report, flat_preds, flat_labels, all_preds, all_labels


def main():
    # 1. Khởi tạo các class cấu hình
    tf_config = TransformerConfig()
    lstm_config = LSTMConfig()
    kd_config = KDConfig()  # Thêm config cho Distillation

    # 2. Khởi tạo Argument Parser
    parser = argparse.ArgumentParser(
        description="Vietnamese Named Entity Recognition orchestrator pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phobert-lora",
        nargs="+",
        choices=["phobert", "phobert-lora", "lstm", "bilstm", "all"],
        help="Choose model architecture to work with (use 'all' to train/evaluate all models sequentially).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "infer", "distill", "quantize"],
        help="Action mode for pipeline execution.",
    )
    parser.add_argument(
        "--use_crf",
        action="store_true",
        default=False,
        help="Attach a Conditional Random Field (CRF) layer to the network output.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override default training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override default training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override default optimizer learning rate.",
    )

    # SỬA LỖI: Thay PATIENCE bằng tf_config.patience
    parser.add_argument(
        "--patience",
        type=int,
        default=tf_config.patience,
        help="Patience epochs for early stopping.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load a specific model weights checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--infer_text",
        type=str,
        default=None,
        help="Vietnamese text sentence for raw inference.",
    )

    # 3. Parse arguments
    args = parser.parse_args()

    batch_size = args.batch_size or tf_config.batch_size

    epochs = args.epochs or (
        tf_config.epochs if "phobert" in args.model else lstm_config.epochs
    )
    learning_rate = args.lr or (
        tf_config.learning_rate
        if "phobert" in args.model
        else lstm_config.learning_rate
    )
    max_seq_length = tf_config.max_seq_length
    val_batch_size = tf_config.val_batch_size

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using compute target device: {device}")

    logger.info("Initializing vinai/phobert-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", keep_accents=True)

    # 1. Mode: TRAIN
    if args.mode == "train":
        logger.info("Loading training and validation datasets...")

        # 1. Đọc dữ liệu raw
        train_sentences = read_conll(TRAIN_FILE)
        val_sentences = read_conll(DEV_FILE)

        # 2. Áp dụng Data Augmentation cho tập Train
        logger.info("Applying Data Augmentation...")
        augmentor = NERDataAugmentor(train_sentences, LABEL2ID)
        augmented_train_sentences = augmentor.generate_augmented_dataset(
            multiplier=1, replace_prob=0.3
        )

        # 3. Khởi tạo Dataset
        train_dataset = VietnameseNERDataset(
            augmented_train_sentences, tokenizer, max_seq_length, LABEL2ID
        )
        val_dataset = VietnameseNERDataset(
            val_sentences, tokenizer, max_seq_length, LABEL2ID
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=tf_config.val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Xác định danh sách model cần train
        if "all" in args.model:
            models_to_train = ["lstm", "bilstm", "phobert", "phobert-lora"]
        else:
            models_to_train = args.model

        # Vòng lặp train lần lượt các model
        for current_model in models_to_train:
            logger.info(
                f"\n{'='*50}\nBẮT ĐẦU TRAIN MODEL: {current_model.upper()}\n{'='*50}"
            )

            # Đặt lại siêu tham số cho từng model cụ thể để tránh lỗi khi args.model == "all"
            current_batch_size = args.batch_size or tf_config.batch_size
            current_epochs = args.epochs or (
                tf_config.epochs if "phobert" in current_model else lstm_config.epochs
            )
            current_lr = args.lr or (
                tf_config.learning_rate
                if "phobert" in current_model
                else lstm_config.learning_rate
            )

            # 4. Khởi tạo DataLoader cho model hiện tại
            train_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )

            logger.info(f"Initializing {current_model} model for training...")
            model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
            model = model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=current_lr)
            criterion = NERLoss(model)

            model_checkpoint_dir = os.path.join(
                CHECKPOINT_DIR, f"{current_model}_crf_{args.use_crf}"
            )
            os.makedirs(model_checkpoint_dir, exist_ok=True)

            trainer = BaseTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                save_dir=model_checkpoint_dir,
                run_name=f"{current_model}_crf_{args.use_crf}",
            )

            logger.info(f"Starting training loop for {current_model}...")
            results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=current_epochs,
                early_stop=args.patience,
            )
            logger.info(
                f"Training completed. Best model for {current_model} saved at: {results['best_path']}"
            )
    # 2. Mode: EVALUATE
    elif args.mode == "evaluate":
        logger.info("Loading evaluation (test) dataset...")
        test_loader = get_dataloader(
            TEST_FILE,
            tokenizer,
            val_batch_size,
            max_seq_length,
            LABEL2ID,
            shuffle=False,
        )
        test_sentences = read_conll(TEST_FILE)

        if "all" in args.model:
            models_to_eval = ["lstm", "bilstm", "phobert", "phobert-lora"]
        else:
            models_to_eval = args.model

        for current_model in models_to_eval:
            logger.info(
                f"\n{'='*50}\nBẮT ĐẦU ĐÁNH GIÁ MODEL: {current_model.upper()}\n{'='*50}"
            )
            model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)

            # Determine checkpoint path
            chk_path = args.checkpoint
            if not chk_path:
                chk_name = (
                    "student_distilled.pt"
                    if "distilled" in current_model
                    else "model.pt"
                )
                model_checkpoint_dir = os.path.join(
                    CHECKPOINT_DIR, f"{current_model}_crf_{args.use_crf}"
                )
                # Lọc file trong thư mục checkpoint
                if os.path.exists(model_checkpoint_dir):
                    files = [
                        os.path.join(model_checkpoint_dir, f)
                        for f in os.listdir(model_checkpoint_dir)
                        if chk_name in f
                    ]
                    if files:
                        chk_path = sorted(files)[-1]

            # CHỐT CHẶN NGHIÊM NGẶT: Bắt buộc phải có file weights
            if not chk_path or not os.path.exists(chk_path):
                logger.error(
                    f"Error: Không tìm thấy checkpoint cho '{current_model}'. Vui lòng train trước!"
                )
                continue  # Bỏ qua model này, chạy đánh giá model tiếp theo

            logger.info(f"Loading weights from checkpoint: {chk_path}")
            ckpt = torch.load(chk_path, map_location=device)
            state_dict = ckpt.get("model_state", ckpt)
            model.load_state_dict(state_dict)
            model = model.to(device)

            pred_save_path = f"results/predictions_{current_model}.txt"
            (
                token_report,
                entity_report,
                flat_preds,
                flat_labels,
                all_preds,
                all_labels,
            ) = run_evaluation(
                model,
                test_loader,
                tokenizer,
                ID2LABEL,
                test_sentences,
                save_pred_path=pred_save_path,
            )

            logger.info(f"\nToken-Level Evaluation Report ({current_model.upper()}):")
            logger.info(f"Overall Accuracy: {token_report['accuracy']:.4f}")
            logger.info(
                f"Macro Precision: {token_report['macro avg']['precision']:.4f}"
            )
            logger.info(f"Macro Recall: {token_report['macro avg']['recall']:.4f}")
            logger.info(f"Macro F1-Score: {token_report['macro avg']['f1-score']:.4f}")

            logger.info(
                f"\nEntity-Level BIO Evaluation Report ({current_model.upper()}):"
            )
            logger.info(
                f"Overall Precision: {entity_report['overall']['precision']:.4f}"
            )
            logger.info(f"Overall Recall: {entity_report['overall']['recall']:.4f}")
            logger.info(f"Overall F1-Score: {entity_report['overall']['f1-score']:.4f}")

            # Plot evaluation curves and reports
            logger.info("Generating performance evaluation plots...")
            plot_confusion_matrix(
                flat_preds,
                flat_labels,
                LABEL_LIST,
                os.path.join(PLOT_DIR, f"{current_model}_confusion_matrix.png"),
            )
            plot_entity_distribution(
                flat_labels,
                LABEL_LIST,
                os.path.join(PLOT_DIR, f"{current_model}_entity_distribution.png"),
            )

    # 3. Mode: INFER
    elif args.mode == "infer":
        if not args.infer_text:
            logger.error(
                "Error: --infer_text is required when --mode is set to 'infer'."
            )
            return

        if "all" in args.model:
            models_to_infer = ["lstm", "bilstm", "phobert", "phobert-lora"]
        else:
            models_to_infer = args.model

        for current_model in models_to_infer:
            logger.info(
                f"\n{'='*50}\nBẮT ĐẦU INFER VỚI MODEL: {current_model.upper()}\n{'='*50}"
            )
            model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)

            chk_path = args.checkpoint
            if not chk_path:
                files = [
                    os.path.join(CHECKPOINT_DIR, f)
                    for f in os.listdir(CHECKPOINT_DIR)
                    if "model.pt" in f
                ]
                if files:
                    chk_path = sorted(files)[-1]

            # CHỐT CHẶN NGHIÊM NGẶT
            if not chk_path or not os.path.exists(chk_path):
                logger.error(
                    f"Error: Trained checkpoint required for inference on '{current_model}'."
                )
                continue

            logger.info(f"Loading weights from {chk_path}")
            ckpt = torch.load(chk_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            model = model.to(device)
            model.eval()

            logger.info(f"Input sentence: {args.infer_text}")
            segmented_text = ViTokenizer.tokenize(args.infer_text)
            words = segmented_text.split()

            input_ids_list = [tokenizer.cls_token_id]
            word_ids = [None]  # Lưu vết index của từ gốc

            for word_idx, word in enumerate(words):
                tokens = tokenizer.tokenize(word)
                if not tokens:
                    continue
                ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids_list.extend(ids)
                # Đánh dấu sub-words này thuộc về từ (word_idx) nào
                word_ids.extend([word_idx] * len(ids))

            input_ids_list.append(tokenizer.sep_token_id)
            word_ids.append(None)

            # Truncation
            if len(input_ids_list) > max_seq_length:
                input_ids_list = input_ids_list[: max_seq_length - 1] + [
                    tokenizer.sep_token_id
                ]
                word_ids = word_ids[: max_seq_length - 1] + [None]

            # Padding
            padding_len = max_seq_length - len(input_ids_list)
            if padding_len > 0:
                input_ids_list.extend([tokenizer.pad_token_id] * padding_len)
                word_ids.extend([None] * padding_len)

            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)

            with torch.no_grad():
                if hasattr(model, "use_crf") and model.use_crf:
                    mask = input_ids != 1
                    best_paths = model.crf.decode(model(input_ids), mask)
                    preds = best_paths[0]
                else:
                    logits = model(input_ids)
                    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

            word_preds = ["O"] * len(words)
            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx < len(words):
                    if word_idx != previous_word_idx:
                        tag_id = preds[idx]
                        word_preds[word_idx] = ID2LABEL.get(tag_id, "O")
                    previous_word_idx = word_idx

            logger.info(f"\n--- INFERENCE RESULTS ({current_model.upper()}) ---")
            for word, tag in zip(words, word_preds):
                if tag != "O":
                    logger.info(
                        f"\033[92m{word} [{tag}]\033[0m", extra={"simple": True}
                    )
                else:
                    logger.info(word, extra={"simple": True})
            logger.info("-------------------------")

    # 4. Mode: DISTILL
    elif args.mode == "distill":
        logger.info("=== KNOWLEDGE DISTILLATION ===")
        # Teacher model luôn là PhoBERT, chỉ cần load 1 lần ngoài vòng lặp
        teacher = PhoBERTModel(num_labels=NUM_LABELS, use_crf=False)

        chk_path = args.checkpoint
        if not chk_path:
            files = [
                os.path.join(CHECKPOINT_DIR, f)
                for f in os.listdir(CHECKPOINT_DIR)
                if "phobert_crf_False" in f and "model.pt" in f
            ]
            if files:
                chk_path = sorted(files)[-1]

        # CHỐT CHẶN TEACHER
        if not chk_path or not os.path.exists(chk_path):
            logger.error(
                "Error: Distillation requires a pre-trained Teacher PhoBERT checkpoint. Train 'phobert' first."
            )
            return

        logger.info(f"Loading Teacher model weights from: {chk_path}")
        ckpt = torch.load(chk_path, map_location=device)
        teacher.load_state_dict(ckpt.get("model_state", ckpt))
        teacher = teacher.to(device)

        # Distill thường áp dụng cho model nhỏ, nếu "all" ta duyệt lstm và bilstm
        if "all" in args.model:
            models_to_distill = ["lstm", "bilstm"]
        else:
            models_to_distill = args.model
        # Khởi tạo DataLoader 1 lần dùng chung
        train_loader = get_dataloader(
            TRAIN_FILE, tokenizer, batch_size, max_seq_length, LABEL2ID, shuffle=True
        )
        val_loader = get_dataloader(
            DEV_FILE, tokenizer, val_batch_size, max_seq_length, LABEL2ID, shuffle=False
        )

        for current_model in models_to_distill:
            logger.info(
                f"\n{'='*50}\nBẮT ĐẦU DISTILL VÀO STUDENT MODEL: {current_model.upper()}\n{'='*50}"
            )
            student = get_model(
                current_model, tokenizer.vocab_size, use_crf=args.use_crf
            )
            student = student.to(device)

            optimizer = optim.AdamW(student.parameters(), lr=lstm_config.learning_rate)
            criterion = NERLoss(student)

            trainer = DistillationTrainer(
                student_model=student,
                teacher_model=teacher,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                alpha=kd_config.alpha,
                temperature=kd_config.temperature,
                save_dir=CHECKPOINT_DIR,
                run_name=f"student_distilled_{current_model}",
            )

            logger.info(f"Starting Knowledge Distillation for {current_model}...")
            results = trainer.train(train_loader, val_loader, epochs=epochs)
            logger.info(
                f"Distillation finished. Best checkpoint for {current_model} saved at: {results['best_path']}"
            )

    # 5. Mode: QUANTIZE
    elif args.mode == "quantize":
        logger.info("=== MODEL COMPRESSION (QUANTIZATION) ===")

        # Lượng tử hóa thường nhắm vào LSTM chạy CPU
        models_to_quantize = ["lstm", "bilstm"] if args.model == "all" else [args.model]

        for current_model in models_to_quantize:
            logger.info(
                f"\n{'='*50}\nBẮT ĐẦU QUANTIZE MODEL: {current_model.upper()}\n{'='*50}"
            )
            model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)

            chk_path = args.checkpoint
            if not chk_path:
                model_checkpoint_dir = os.path.join(
                    CHECKPOINT_DIR, f"{current_model}_crf_{args.use_crf}"
                )

                if os.path.exists(model_checkpoint_dir):
                    # Tìm tất cả file .pt trong thư mục (loại trừ các file đã quantize trước đó)
                    files = [
                        os.path.join(model_checkpoint_dir, f)
                        for f in os.listdir(model_checkpoint_dir)
                        if f.endswith(".pt") and "quantized" not in f
                    ]
                    if files:
                        # Lấy file mới nhất
                        chk_path = sorted(files)[-1]

            # CHỐT CHẶN NGHIÊM NGẶT
            if not chk_path or not os.path.exists(chk_path):
                logger.error(
                    f"Error: Quantization requires a trained checkpoint for '{current_model}'."
                )
                continue

            logger.info(f"Loading weights from baseline: {chk_path}")
            ckpt = torch.load(chk_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state", ckpt))
            model = model.to("cpu")

            logger.info("Performing Post-Training Dynamic Quantization (PTQ)...")
            quantized_model = quantize_utils.quantize_dynamic_ptq(model)

            q_save_path = os.path.join(
                CHECKPOINT_DIR, f"{current_model}_quantized_ptq.pt"
            )
            torch.save(quantized_model.state_dict(), q_save_path)
            logger.info(f"Quantized dynamic PTQ model saved at: {q_save_path}")

            val_loader = get_dataloader(
                DEV_FILE,
                tokenizer,
                val_batch_size,
                max_seq_length,
                LABEL2ID,
                shuffle=False,
            )

            logger.info(
                f"Evaluating quantized {current_model} on Validation set (CPU)..."
            )
            quantized_model = quantized_model.cpu()

            token_report, entity_report, _, _, _, _ = run_evaluation(
                quantized_model, val_loader, tokenizer, ID2LABEL, None
            )
            logger.info(
                f"Quantized Model ({current_model}) Accuracy: {token_report['accuracy']:.4f}"
            )
            logger.info(
                f"Quantized Model ({current_model}) Entity F1-Score: {entity_report['overall']['f1-score']:.4f}"
            )


# Đảm bảo phần dưới đây là phần cuối cùng của file main.py
if __name__ == "__main__":
    main()
