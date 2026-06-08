import os
import argparse
import logging
import torch
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from src import *
from torch.utils.data import DataLoader
import src.quantize_utils as quantize_utils
import transformers

transformers.utils.logging.set_verbosity_error()
# Set up main logger
logger = setup_logger("main_orchestrator")


def main():
    # 0. Set seed for reproducibility
    set_seed(seed=42)

    # 1. Initialize configuration classes
    tf_config = TransformerConfig()
    bert_config = BERTConfig()
    lstm_config = LSTMConfig()
    kd_config = KDConfig()  # Add configuration for Distillation

    # 2. Initialize Argument Parser
    parser = argparse.ArgumentParser(
        description="Vietnamese Named Entity Recognition orchestrator pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        nargs="+",
        choices=["phobert", "phobert-lora", "transformer", "lstm", "bilstm", "all"],
        help="Choose model architecture to work with (use 'all' to train/evaluate all models sequentially).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "infer", "distill", "quantize", "train_qat"],
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

    # FIX: Replace PATIENCE with tf_config.patience
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

    parser.add_argument(
        "--force_cpu_crf",
        action="store_true",
        default=False,
        help="Force CPU usage when using CRF (especially on Apple Silicon) to avoid unsupported operations.",
    )
    # 3. Parse command-line arguments
    args = parser.parse_args()

    if "all" in args.model:
        logger.info("Chế độ huấn luyện/đánh giá toàn bộ các mô hình tuần tự.")
        base_cfg = None
    else:
        primary_model = args.model[0]
        if "phobert" in primary_model:
            base_cfg = bert_config
        elif primary_model == "transformer":
            base_cfg = tf_config
        else:
            base_cfg = lstm_config

    # Only compute these fallback values when not using the 'all' mode
    batch_size = args.batch_size or (base_cfg.batch_size if base_cfg else None)
    epochs = args.epochs or (base_cfg.epochs if base_cfg else None)
    learning_rate = args.lr or (base_cfg.learning_rate if base_cfg else None)
    max_seq_length = base_cfg.max_seq_length if base_cfg else None
    val_batch_size = base_cfg.val_batch_size if base_cfg else None

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # --- FIX: Safe fallback for CRF on MPS (Apple Silicon) ---
    if args.use_crf and (device.type == "mps" or args.force_cpu_crf):
        logger.warning(
            "Detected CRF usage on MPS (Apple Silicon) or the --force_cpu_crf flag is enabled. "
            "Forcing fallback to CPU to prevent asynchronous `torch.logsumexp` errors."
        )
        device = torch.device("cpu")

    logger.info(f"Using compute target device: {device}")

    logger.info("Initializing vinai/phobert-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", keep_accents=True)

    if args.mode == "train":
        run_train(
            args, bert_config, tf_config, lstm_config, tokenizer, device, LABEL2ID
        )

    elif args.mode == "evaluate":
        run_evaluate(
            args,
            bert_config,
            tf_config,
            lstm_config,
            tokenizer,
            device,
            LABEL2ID,
            ID2LABEL,
            LABEL_LIST,
        )

    elif args.mode == "infer":
        run_infer(
            args, bert_config, tf_config, lstm_config, tokenizer, device, ID2LABEL
        )

    elif args.mode == "distill":
        run_distill(
            args,
            bert_config,
            tf_config,
            lstm_config,
            kd_config,
            tokenizer,
            device,
            LABEL2ID,
            NUM_LABELS,
        )

    elif args.mode == "quantize":
        run_quantize(
            args,
            bert_config,
            tf_config,
            lstm_config,
            tokenizer,
            device,
            LABEL2ID,
            ID2LABEL,
        )

    elif args.mode == "train_qat":
        run_train_qat(
            args, bert_config, tf_config, lstm_config, tokenizer, device, LABEL2ID
        )
    else:
        logger.error(f"Unknown mode: {args.mode}")


# Ensure the following block is the script entry point
if __name__ == "__main__":
    main()
