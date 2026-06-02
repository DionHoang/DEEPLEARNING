import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
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

    if args.mode == "train":
        run_train(args, tf_config, lstm_config, tokenizer, device, LABEL2ID)

    elif args.mode == "evaluate":
        run_evaluate(args, tf_config, tokenizer, device, LABEL2ID, ID2LABEL, LABEL_LIST)

    elif args.mode == "infer":
        run_infer(args, tf_config, tokenizer, device, ID2LABEL)

    elif args.mode == "distill":
        run_distill(
            args,
            tf_config,
            lstm_config,
            kd_config,
            tokenizer,
            device,
            LABEL2ID,
            NUM_LABELS,
        )

    elif args.mode == "quantize":
        run_quantize(args, tf_config, tokenizer, device, LABEL2ID, ID2LABEL)

    else:
        logger.error(f"Unknown mode: {args.mode}")


# Đảm bảo phần dưới đây là phần cuối cùng của file main.py
if __name__ == "__main__":
    main()
