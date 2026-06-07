import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from src import *
import src.quantize_utils as quantize_utils

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transformers

transformers.utils.logging.set_verbosity_error()  # Suppress non-essential transformers logs

logger = setup_logger("engine")


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
                    padded_preds = preds + [-100] * (len(gold_labels) - len(preds))

                    all_preds.append(padded_preds)
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


def run_train(args, bert_config, tf_config, lstm_config, tokenizer, device, LABEL2ID):
    logger.info("Loading training and validation datasets...")

    # Read raw CoNLL-formatted data files
    train_sentences = read_conll(TRAIN_FILE)
    val_sentences = read_conll(DEV_FILE)

    # Apply data augmentation to the training set
    logger.info("Applying Data Augmentation...")
    augmentor = NERDataAugmentor(train_sentences, LABEL2ID)
    augmented_train_sentences = augmentor.generate_augmented_dataset(
        multiplier=1, replace_prob=0.3
    )

    # Determine which models to train
    if "all" in args.model:
        models_to_train = ["lstm", "bilstm", "transformer", "phobert", "phobert-lora"]
    else:
        models_to_train = args.model

    # Loop over models to train them sequentially
    for current_model in models_to_train:
        logger.info(
            f"\n{'='*50}\nBẮT ĐẦU TRAIN MODEL: {current_model.upper()}\n{'='*50}"
        )

        if "phobert" in current_model:
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        current_max_len = active_cfg.max_seq_length
        current_batch_size = args.batch_size or active_cfg.batch_size
        current_val_batch_size = active_cfg.val_batch_size
        current_epochs = args.epochs or active_cfg.epochs
        current_lr = args.lr or active_cfg.learning_rate

        # Initialize Dataset & DataLoader according to the active model configuration
        train_dataset = VietnameseNERDataset(
            augmented_train_sentences, tokenizer, current_max_len, LABEL2ID
        )
        val_dataset = VietnameseNERDataset(
            val_sentences, tokenizer, current_max_len, LABEL2ID
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        logger.info(f"Initializing {current_model} model for training...")
        model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        model = model.to(device)
        if current_model == "phobert-lora":
            lora_params, classifier_params, crf_params = [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "classifier" in name:
                    classifier_params.append(param)
                elif "crf" in name:
                    crf_params.append(param)
                else:
                    lora_params.append(param)

            param_groups = [
                {"params": lora_params, "lr": 2e-4},
                {"params": classifier_params, "lr": 1e-3},
            ]
            if crf_params:
                param_groups.append({"params": crf_params, "lr": 1e-3})
            optimizer = optim.AdamW(param_groups)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

            from transformers import get_linear_schedule_with_warmup

            total_steps = len(train_loader) * current_epochs
            warmup_steps = int(total_steps * LoRAConfig().warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        criterion = NERLoss(model)

        dirs = get_model_dirs(current_model, args.use_crf)
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            save_dir=dirs["checkpoints"],
            tensorboard_dir=dirs["tensorboard"],
            log_dir=dirs["logs"],
            run_name=f"{current_model}_crf_{args.use_crf}",
        )

        start_epoch = 1
        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info(f"Restoring state from checkpoint: {args.checkpoint}")
            ckpt = trainer.load_checkpoint(args.checkpoint)
            start_epoch = ckpt.get("epoch", 0) + 1

        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=current_epochs,
            early_stop=args.patience,
            start_epoch=start_epoch,
        )
        logger.info(
            f"Training completed. Best model for {current_model} saved at: {results['best_path']}"
        )

        history = results.get("history")
        if history:
            logger.info(f"Generating training plots for {current_model}...")
            plot_loss_curve(
                history,
                save_path=os.path.join(
                    dirs["plots"], f"{current_model}_loss_curve.png"
                ),
            )
            plot_acc_curve(
                history,
                save_path=os.path.join(dirs["plots"], f"{current_model}_acc_curve.png"),
            )


def run_evaluate(
    args,
    bert_config,
    tf_config,
    lstm_config,
    tokenizer,
    device,
    LABEL2ID,
    ID2LABEL,
    LABEL_LIST,
):
    logger.info("Loading evaluation (test) dataset...")
    test_sentences = read_conll(TEST_FILE)

    if "all" in args.model:
        models_to_eval = ["lstm", "bilstm", "transformer", "phobert", "phobert-lora"]
    else:
        models_to_eval = args.model

    for current_model in models_to_eval:
        logger.info(
            f"\n{'='*50}\nBẮT ĐẦU ĐÁNH GIÁ MODEL: {current_model.upper()}\n{'='*50}"
        )

        if "phobert" in current_model:
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        test_loader = get_dataloader(
            TEST_FILE,
            tokenizer,
            active_cfg.val_batch_size,
            active_cfg.max_seq_length,
            LABEL2ID,
            shuffle=False,
        )

        model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        dirs = get_model_dirs(current_model, args.use_crf)

        chk_path = args.checkpoint
        if not chk_path:
            model_checkpoint_dir = dirs["checkpoints"]
            if os.path.exists(model_checkpoint_dir):
                files = [
                    os.path.join(model_checkpoint_dir, f)
                    for f in os.listdir(model_checkpoint_dir)
                    if "model.pt" in f
                ]
                if files:
                    chk_path = sorted(files)[-1]

        if not chk_path or not os.path.exists(chk_path):
            logger.error(f"Error: Could not find checkpoint for '{current_model}'.")
            continue

        logger.info(f"Loading weights from checkpoint: {chk_path}")
        ckpt = torch.load(chk_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model = model.to(device)

        pred_save_path = os.path.join(dirs["base"], "predictions.txt")
        token_report, entity_report, flat_preds, flat_labels, _, _ = run_evaluation(
            model,
            test_loader,
            tokenizer,
            ID2LABEL,
            test_sentences,
            save_pred_path=pred_save_path,
        )

        logger.info(f"\nToken-Level Evaluation Report ({current_model.upper()}):")
        logger.info(f"Overall Accuracy: {token_report['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {token_report['macro avg']['f1-score']:.4f}")
        logger.info(f"\nEntity-Level BIO Evaluation Report ({current_model.upper()}):")
        logger.info(f"Overall F1-Score: {entity_report['overall']['f1-score']:.4f}")
        logger.info(
            f"\nDetailed Entity Report ({current_model.upper()}):\n{entity_report['report']}"
        )
        csv_path = os.path.join(dirs["base"], f"{current_model}_token_metrics.csv")
        save_metrics_csv(token_report, csv_path)

        plot_confusion_matrix(
            flat_preds,
            flat_labels,
            LABEL_LIST,
            os.path.join(dirs["plots"], "confusion_matrix.png"),
        )
        plot_entity_distribution(
            flat_labels,
            LABEL_LIST,
            os.path.join(dirs["plots"], "entity_distribution.png"),
        )


def run_infer(args, bert_config, tf_config, lstm_config, tokenizer, device, ID2LABEL):
    """Run inference supporting all model variants including the pure Transformer."""
    if not args.infer_text:
        logger.error("Error: --infer_text is required when --mode is set to 'infer'.")
        return

    if "all" in args.model:
        models_to_infer = ["lstm", "bilstm", "transformer", "phobert", "phobert-lora"]
    else:
        models_to_infer = args.model

    for current_model in models_to_infer:
        logger.info(
            f"\n{'='*50}\nBẮT ĐẦU INFER VỚI MODEL: {current_model.upper()}\n{'='*50}"
        )

        # Dynamically select model configuration to avoid mismatched max_length
        if "phobert" in current_model:
            active_cfg = bert_config
        elif current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        max_seq_length = active_cfg.max_seq_length

        model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        dirs = get_model_dirs(current_model, args.use_crf)

        chk_path = args.checkpoint
        if not chk_path:
            model_checkpoint_dir = dirs["checkpoints"]
            if os.path.exists(model_checkpoint_dir):
                if "best_model.pt" in os.listdir(model_checkpoint_dir):
                    chk_path = os.path.join(model_checkpoint_dir, "best_model.pt")
                else:
                    files = [
                        os.path.join(model_checkpoint_dir, f)
                        for f in os.listdir(model_checkpoint_dir)
                        if "model.pt" in f and "best" not in f
                    ]
                    if files:
                        chk_path = sorted(files)[-1]

        if not chk_path or not os.path.exists(chk_path):
            logger.error(
                f"Error: Trained checkpoint required for inference on '{current_model}'."
            )
            continue

        is_quantized = "quantized" in chk_path

        if is_quantized:
            logger.info(
                "Quantized model detected. Forcing CPU fallback for compatibility."
            )
            infer_device = torch.device("cpu")
            logger.info(f"Loading full quantized object from {chk_path}")
            # Load the full quantized object directly
            model = torch.load(chk_path, map_location=infer_device, weights_only=False)
        else:
            infer_device = device
            logger.info(f"Loading weights from {chk_path}")
            ckpt = torch.load(chk_path, map_location=infer_device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state", ckpt))

        model = model.to(infer_device)
        model.eval()

        logger.info(f"Input sentence: {args.infer_text}")
        segmented_text = ViTokenizer.tokenize(args.infer_text)
        words = segmented_text.split()

        input_ids_list = [tokenizer.cls_token_id]
        word_ids = [None]  # Track which original word each subword belongs to

        for word_idx, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            if not tokens:
                continue
            ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids_list.extend(ids)
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

        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(infer_device)

        with torch.no_grad():
            decoded_output = model.decode(input_ids)
            if not decoded_output:
                logger.warning("Model returned empty decode output.")
                continue
            preds = decoded_output[0]

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
                print(f"\033[92m{word} [{tag}]\033[0m")
            else:
                print(word)
        logger.info("-------------------------")


def run_distill(
    args,
    bert_config,
    tf_config,
    lstm_config,
    kd_config,
    tokenizer,
    device,
    LABEL2ID,
    NUM_LABELS,
):
    """Multi-architecture knowledge distillation: distill PhoBERT into LSTM/Transformer students."""
    logger.info("=== KNOWLEDGE DISTILLATION ===")

    # The teacher model is fixed to the PhoBERT-base configuration
    teacher = PhoBERTModel(num_labels=NUM_LABELS, use_crf=args.use_crf)
    teacher_dirs = get_model_dirs("phobert", args.use_crf)

    chk_path = args.checkpoint
    if not chk_path:
        teacher_ckpt_dir = teacher_dirs["checkpoints"]
        if os.path.exists(teacher_ckpt_dir):
            files = [
                os.path.join(teacher_ckpt_dir, f)
                for f in os.listdir(teacher_ckpt_dir)
                if "model.pt" in f
            ]
            if files:
                chk_path = sorted(files)[-1]

    if not chk_path or not os.path.exists(chk_path):
        logger.error(
            "Error: Distillation requires a pre-trained Teacher PhoBERT checkpoint. Train 'phobert' first."
        )
        return

    logger.info(f"Loading Teacher model weights from: {chk_path}")
    ckpt = torch.load(chk_path, map_location=device)
    teacher.load_state_dict(ckpt.get("model_state", ckpt))
    teacher = teacher.to(device)

    # Configure which student models to distill into
    if "all" in args.model:
        models_to_distill = ["lstm", "bilstm", "transformer"]
    else:
        models_to_distill = args.model

    for current_model in models_to_distill:
        logger.info(
            f"\n{'='*50}\nBẮT ĐẦU DISTILL VÀO STUDENT MODEL: {current_model.upper()}\n{'='*50}"
        )

        # Configure student-specific hyperparameters
        if current_model == "transformer":
            student_cfg = tf_config
        else:
            student_cfg = lstm_config

        max_seq_length = student_cfg.max_seq_length
        val_batch_size = student_cfg.val_batch_size
        batch_size = args.batch_size or student_cfg.batch_size
        epochs = args.epochs or student_cfg.epochs
        learning_rate = student_cfg.learning_rate

        train_loader = get_dataloader(
            TRAIN_FILE, tokenizer, batch_size, max_seq_length, LABEL2ID, shuffle=True
        )
        val_loader = get_dataloader(
            DEV_FILE, tokenizer, val_batch_size, max_seq_length, LABEL2ID, shuffle=False
        )

        student = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        student = student.to(device)

        # Use student-specific optimizer and hyperparameters as configured
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, student.parameters()), lr=learning_rate
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        criterion = NERLoss(student)

        student_dirs = get_model_dirs(current_model, args.use_crf)

        trainer = DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            alpha=kd_config.alpha,
            temperature=kd_config.temperature,
            save_dir=student_dirs["checkpoints"],
            tensorboard_dir=student_dirs["tensorboard"],
            log_dir=student_dirs["logs"],
            run_name=f"student_distilled_{current_model}_crf_{args.use_crf}",
        )

        logger.info(f"Starting Knowledge Distillation for Student: {current_model}...")
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        logger.info(
            f"Distillation finished. Best checkpoint for {current_model} saved at: {results['best_path']}"
        )


def run_quantize(
    args, bert_config, tf_config, lstm_config, tokenizer, device, LABEL2ID, ID2LABEL
):
    """Perform dynamic PTQ quantization, with added support for linear-heavy models like Transformer."""
    logger.info("=== MODEL COMPRESSION (QUANTIZATION) ===")

    # Include transformer in the list of models suitable for linear-based quantization
    if "all" in args.model:
        models_to_quantize = ["lstm", "bilstm", "transformer"]
    else:
        models_to_quantize = args.model

    for current_model in models_to_quantize:
        if current_model == "phobert" or current_model == "phobert-lora":
            logger.warning(
                f"Skipping {current_model}. Large Transformer models require specialized quantization (e.g. bitsandbytes 4-bit)."
            )
            continue

        logger.info(
            f"\n{'='*50}\nBẮT ĐẦU QUANTIZE MODEL: {current_model.upper()}\n{'='*50}"
        )

        if current_model == "transformer":
            active_cfg = tf_config
        else:
            active_cfg = lstm_config

        max_seq_length = active_cfg.max_seq_length
        val_batch_size = active_cfg.val_batch_size

        model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)
        dirs = get_model_dirs(current_model, args.use_crf)

        chk_path = args.checkpoint
        if not chk_path:
            model_checkpoint_dir = dirs["checkpoints"]
            if os.path.exists(model_checkpoint_dir):
                files = [
                    os.path.join(model_checkpoint_dir, f)
                    for f in os.listdir(model_checkpoint_dir)
                    if f.endswith(".pt") and "quantized" not in f
                ]
                if files:
                    chk_path = sorted(files)[-1]

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
        # Quantize Linear & LSTM layers inside the model to INT8
        if current_model == "transformer":
            logger.warning(
                "Applying PTQ bypass for Transformer to avoid 'device' attribute bug."
            )
            ops_to_quantize = {
                nn.Linear
            }  # Transformer không có LSTM, giúp bypass an toàn
        else:
            ops_to_quantize = {nn.Linear, nn.LSTM}

        quantized_model = quantize_utils.quantize_dynamic_ptq(
            model, operators_to_quantize=ops_to_quantize
        )

        q_save_path = os.path.join(
            dirs["checkpoints"], f"{current_model}_quantized_ptq.pt"
        )
        torch.save(quantized_model, q_save_path)
        logger.info(f"Quantized dynamic PTQ model saved at: {q_save_path}")

        val_loader = get_dataloader(
            DEV_FILE, tokenizer, val_batch_size, max_seq_length, LABEL2ID, shuffle=False
        )

        logger.info(f"Evaluating quantized {current_model} on Validation set (CPU)...")
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


def run_train_qat(
    args, bert_config, tf_config, lstm_config, tokenizer, device, LABEL2ID
):
    logger.info("=== STARTING QUANTIZATION AWARE TRAINING (QAT) ===")

    train_sentences = read_conll(TRAIN_FILE)
    val_sentences = read_conll(DEV_FILE)

    if "all" in args.model:
        models_to_train = ["lstm", "bilstm", "transformer"]
    else:
        models_to_train = args.model

    for current_model in models_to_train:
        if current_model in ["phobert", "phobert-lora"]:
            logger.warning(
                f"Skipping QAT for {current_model}. HF models need specific int8 configs."
            )
            continue

        logger.info(f"\n{'='*50}\nBẮT ĐẦU QAT MODEL: {current_model.upper()}\n{'='*50}")

        active_cfg = tf_config if current_model == "transformer" else lstm_config

        current_epochs = args.epochs or active_cfg.epochs
        current_batch_size = args.batch_size or active_cfg.batch_size

        train_loader = get_dataloader(
            TRAIN_FILE,
            tokenizer,
            current_batch_size,
            active_cfg.max_seq_length,
            LABEL2ID,
            shuffle=True,
        )
        val_loader = get_dataloader(
            DEV_FILE,
            tokenizer,
            active_cfg.val_batch_size,
            active_cfg.max_seq_length,
            LABEL2ID,
            shuffle=False,
        )

        model = get_model(current_model, tokenizer.vocab_size, use_crf=args.use_crf)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=active_cfg.learning_rate,
        )
        criterion = NERLoss(model)
        dirs = get_model_dirs(
            f"{current_model}_qat", args.use_crf
        )  # Lưu riêng thư mục qat

        trainer = QuantizationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            quantize_utils=quantize_utils,
            save_dir=dirs["checkpoints"],
            tensorboard_dir=dirs["tensorboard"],
            log_dir=dirs["logs"],
            run_name=f"{current_model}_qat_crf_{args.use_crf}",
            force_convert_to_cpu=True,
        )

        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=current_epochs,
            early_stop=args.patience,
        )
        logger.info(
            f"QAT completed for {current_model.upper()}. Quantized model ready for deployment."
        )
