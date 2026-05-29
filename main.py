<<<<<<< Updated upstream
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from src.config import (
    TRAIN_FILE, DEV_FILE, TEST_FILE, LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS,
    MAX_SEQ_LENGTH, BATCH_SIZE, VAL_BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    LSTM_EMBEDDING_DIM, LSTM_HIDDEN_DIM, LSTM_DROPOUT, LSTM_LEARNING_RATE, LSTM_EPOCHS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, KD_TEMPERATURE, KD_ALPHA, CHECKPOINT_DIR, PLOT_DIR
)
from src.dataset import get_dataloader, read_conll
from src.model import LSTMModel, BiLSTMModel, PhoBERTModel, PhoBERTLoRA
from src.trainer import BaseTrainer, DistillationTrainer, QuantizationTrainer
import src.quantize_utils as quantize_utils
from src.utils import (
    setup_logger, compute_entity_level_metrics, compute_metrics,
    plot_loss_curve, plot_f1_curve, plot_confusion_matrix,
    plot_entity_distribution, save_metrics_csv, save_predictions,
    get_error_analysis
)

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

    def forward(self, outputs, labels):
        if hasattr(self.model, "use_crf") and self.model.use_crf:
            return self.model.crf_loss(outputs, labels)
        else:
            return self.ce(outputs.view(-1, outputs.shape[-1]), labels.view(-1))


def get_model(model_name, vocab_size, use_crf=False):
    """Instantiate the requested model class."""
    if model_name == "lstm":
        logger.info(f"Instantiating LSTM model (CRF={use_crf})")
        return LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=LSTM_EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_labels=NUM_LABELS,
            dropout=LSTM_DROPOUT,
            use_crf=use_crf
        )
    elif model_name == "bilstm":
        logger.info(f"Instantiating Bidirectional LSTM model (CRF={use_crf})")
        return BiLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=LSTM_EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_labels=NUM_LABELS,
            dropout=LSTM_DROPOUT,
            use_crf=use_crf
        )
    elif model_name == "phobert":
        logger.info(f"Instantiating PhoBERT model (CRF={use_crf})")
        return PhoBERTModel(num_labels=NUM_LABELS, use_crf=use_crf)
    elif model_name == "phobert-lora":
        logger.info(f"Instantiating PhoBERT + LoRA model (CRF={use_crf})")
        return PhoBERTLoRA(num_labels=NUM_LABELS, use_crf=use_crf)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_evaluation(model, dataloader, tokenizer, id2label, dataset_sentences, save_pred_path=None):
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
                    preds = preds[:active_len] + [-100] * (len(gold_labels) - len(preds[:active_len]))
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
            t_seq_clean = [t for t in l_seq if t != -100]
            p_seq_clean = p_seq[:len(t_seq_clean)]
            clean_trues.append(t_seq_clean)
            clean_preds.append(p_seq_clean)
            
        save_predictions(tokens_list, clean_trues, clean_preds, save_pred_path, id2label)

    return token_report, entity_report, flat_preds, flat_labels, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Named Entity Recognition orchestrator pipeline.")
    parser.add_argument("--model", type=str, default="phobert", choices=["phobert", "phobert-lora", "lstm", "bilstm"],
                        help="Choose model architecture to work with.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "infer", "distill", "quantize"],
                        help="Action mode for pipeline execution.")
    parser.add_argument("--use_crf", action="store_true", default=False,
                        help="Attach a Conditional Random Field (CRF) layer to the network output.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override default training batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override default optimizer learning rate.")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Patience epochs for early stopping.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load a specific model weights checkpoint (.pt) file.")
    parser.add_argument("--infer_text", type=str, default=None,
                        help="Vietnamese text sentence for raw inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using compute target device: {device}")

    # Set up tokenizer (VinAI PhoBERT-base is the project standard)
    logger.info("Initializing vinai/phobert-base tokenizer...")
    # Add keep_accents=True for Vietnamese characters correctness
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", keep_accents=True)

    # Load dataset dataloaders
    batch_size = args.batch_size or (BATCH_SIZE if "phobert" in args.model else BATCH_SIZE)
    epochs = args.epochs or (EPOCHS if "phobert" in args.model else LSTM_EPOCHS)
    learning_rate = args.lr or (LEARNING_RATE if "phobert" in args.model else LSTM_LEARNING_RATE)

    # 1. Mode: TRAIN
    if args.mode == "train":
        logger.info("Loading training and validation datasets...")
        train_loader = get_dataloader(TRAIN_FILE, tokenizer, batch_size, MAX_SEQ_LENGTH, LABEL2ID, shuffle=True)
        val_loader = get_dataloader(DEV_FILE, tokenizer, VAL_BATCH_SIZE, MAX_SEQ_LENGTH, LABEL2ID, shuffle=False)

        model = get_model(args.model, tokenizer.vocab_size, use_crf=args.use_crf)
        model = model.to(device)

        criterion = NERLoss(model)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        run_name = f"{args.model}_crf_{args.use_crf}_lr_{learning_rate}"
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            save_dir=CHECKPOINT_DIR,
            run_name=run_name
        )

        logger.info("Starting training loop...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            grad_clip=1.0,
            early_stop=args.patience
        )
        logger.info(f"Training completed successfully. Best validation loss: {results['best_val']:.4f}")
        logger.info(f"Best checkpoint saved at: {results['best_path']}")

    # 2. Mode: EVALUATE
    elif args.mode == "evaluate":
        logger.info("Loading evaluation (test) dataset...")
        test_loader = get_dataloader(TEST_FILE, tokenizer, VAL_BATCH_SIZE, MAX_SEQ_LENGTH, LABEL2ID, shuffle=False)
        test_sentences = read_conll(TEST_FILE)

        model = get_model(args.model, tokenizer.vocab_size, use_crf=args.use_crf)
        
        # Determine checkpoint path
        chk_path = args.checkpoint
        if not chk_path:
            chk_name = "student_distilled.pt" if "distilled" in args.model else "model.pt"
            # Find the latest matching file in checkpoint dir
            files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if chk_name in f]
            if files:
                chk_path = sorted(files)[-1]
            
        if not chk_path or not os.path.exists(chk_path):
            logger.warning(f"No checkpoint found matching {chk_path or 'default'}. Running evaluation with random weights!")
        else:
            logger.info(f"Loading weights from checkpoint: {chk_path}")
            ckpt = torch.load(chk_path, map_location=device)
            # Support loading state dictionary from trainer state
            state_dict = ckpt.get("model_state", ckpt)
            model.load_state_dict(state_dict)

        model = model.to(device)
        
        pred_save_path = "results/predictions.txt"
        token_report, entity_report, flat_preds, flat_labels, all_preds, all_labels = run_evaluation(
            model, test_loader, tokenizer, ID2LABEL, test_sentences, save_pred_path=pred_save_path
        )

        logger.info("\nToken-Level Evaluation Report:")
        logger.info(f"Overall Accuracy: {token_report['accuracy']:.4f}")
        logger.info(f"Macro Precision: {token_report['macro avg']['precision']:.4f}")
        logger.info(f"Macro Recall: {token_report['macro avg']['recall']:.4f}")
        logger.info(f"Macro F1-Score: {token_report['macro avg']['f1-score']:.4f}")

        logger.info("\nEntity-Level BIO Evaluation Report:")
        logger.info(f"Overall Precision: {entity_report['overall']['precision']:.4f}")
        logger.info(f"Overall Recall: {entity_report['overall']['recall']:.4f}")
        logger.info(f"Overall F1-Score: {entity_report['overall']['f1-score']:.4f}")

        # Render Per-Entity metrics
        logger.info("\nPer-Entity type metrics:")
        for etype, metrics in entity_report["per_type"].items():
            logger.info(f"  {etype:<25} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f} | Support: {metrics['support']}")

        # Plot evaluation curves and reports
        logger.info("Generating performance evaluation plots...")
        plot_confusion_matrix(flat_preds, flat_labels, LABEL_LIST, os.path.join(PLOT_DIR, f"{args.model}_confusion_matrix.png"))
        plot_entity_distribution(flat_labels, LABEL_LIST, os.path.join(PLOT_DIR, "entity_distribution.png"))
        
        # Error Analysis print
        logger.info("Performing Error Analysis sample display...")
        raw_words = [sent["words"] for sent in test_sentences]
        get_error_analysis(raw_words, all_labels, all_preds, ID2LABEL, num_samples=5)

    # 3. Mode: INFER
    elif args.mode == "infer":
        if not args.infer_text:
            logger.error("Error: --infer_text is required when --mode is set to 'infer'.")
            return

        model = get_model(args.model, tokenizer.vocab_size, use_crf=args.use_crf)
        chk_path = args.checkpoint
        if not chk_path:
            files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if "model.pt" in f]
            if files:
                chk_path = sorted(files)[-1]

        if not chk_path or not os.path.exists(chk_path):
            logger.error("Error: Trained model checkpoint is required for inference.")
            return

        logger.info(f"Loading weights from {chk_path}")
        ckpt = torch.load(chk_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model = model.to(device)
        model.eval()

        logger.info(f"Input sentence: {args.infer_text}")
        
        # Word segment or split by spaces for Vietnamese
        words = args.infer_text.split()
        
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        word_ids = encoding.word_ids()

        with torch.no_grad():
            if hasattr(model, "use_crf") and model.use_crf:
                mask = (input_ids != 1)
                best_paths = model.crf.decode(model(input_ids), mask)
                preds = best_paths[0]
            else:
                logits = model(input_ids)
                preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # Map token predictions back to original words (only first subwords)
        word_preds = ["O"] * len(words)
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(words):
                if word_idx != previous_word_idx:
                    tag_id = preds[idx]
                    word_preds[word_idx] = ID2LABEL.get(tag_id, "O")
                previous_word_idx = word_idx

        # Render colored text highlight of named entities in console
        logger.info("\n--- INFERENCE RESULTS ---")
        for word, tag in zip(words, word_preds):
            if tag != "O":
                logger.info(f"\033[92m{word} [{tag}]\033[0m", extra={"simple": True})
            else:
                logger.info(word, extra={"simple": True})
        logger.info("-------------------------")

    # 4. Mode: DISTILL
    elif args.mode == "distill":
        logger.info("=== KNOWLEDGE DISTILLATION ===")
        # Teacher model: pretrained PhoBERT
        teacher = PhoBERTModel(num_labels=NUM_LABELS, use_crf=False)
        chk_path = args.checkpoint
        if not chk_path:
            # Look for best PhoBERT checkpoint
            files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if "phobert_crf_False" in f and "model.pt" in f]
            if files:
                chk_path = sorted(files)[-1]
                
        if not chk_path or not os.path.exists(chk_path):
            logger.error("Error: Distillation requires a pre-trained Teacher PhoBERT checkpoint. Train 'phobert' first.")
            return

        logger.info(f"Loading Teacher model weights from: {chk_path}")
        ckpt = torch.load(chk_path, map_location=device)
        teacher.load_state_dict(ckpt.get("model_state", ckpt))
        teacher = teacher.to(device)

        # Student model: LSTM or BiLSTM (smaller baseline)
        student = get_model(args.model, tokenizer.vocab_size, use_crf=args.use_crf)
        student = student.to(device)

        # Load loaders
        train_loader = get_dataloader(TRAIN_FILE, tokenizer, batch_size, MAX_SEQ_LENGTH, LABEL2ID, shuffle=True)
        val_loader = get_dataloader(DEV_FILE, tokenizer, VAL_BATCH_SIZE, MAX_SEQ_LENGTH, LABEL2ID, shuffle=False)

        optimizer = optim.AdamW(student.parameters(), lr=LSTM_LEARNING_RATE)
        criterion = NERLoss(student)

        trainer = DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            alpha=KD_ALPHA,
            temperature=KD_TEMPERATURE,
            save_dir=CHECKPOINT_DIR,
            run_name=f"student_distilled_{args.model}"
        )

        logger.info("Starting Knowledge Distillation training...")
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        logger.info(f"Distillation finished. Best Student checkpoint saved at: {results['best_path']}")

    # 5. Mode: QUANTIZE
    elif args.mode == "quantize":
        logger.info("=== MODEL COMPRESSION (QUANTIZATION) ===")
        # Quantize support is usually for LSTM-based models on CPU
        model = get_model(args.model, tokenizer.vocab_size, use_crf=args.use_crf)
        
        chk_path = args.checkpoint
        if not chk_path:
            chk_name = "student_distilled.pt" if "distilled" in args.model else "model.pt"
            files = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if chk_name in f]
            if files:
                chk_path = sorted(files)[-1]

        if not chk_path or not os.path.exists(chk_path):
            logger.error("Error: Quantization requires a trained baseline model checkpoint.")
            return

        logger.info(f"Loading weights from baseline: {chk_path}")
        ckpt = torch.load(chk_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        model = model.to(device)

        logger.info("Performing Post-Training Dynamic Quantization (PTQ)...")
        # Quantize Dynamic (Linear, LSTM operators) to 8-bit integers
        quantized_model = quantize_utils.quantize_dynamic_ptq(model)
        
        # Save quantized model
        q_save_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_quantized_ptq.pt")
        torch.save(quantized_model.state_dict(), q_save_path)
        logger.info(f"Quantized dynamic PTQ model saved at: {q_save_path}")

        # If val dataloader is available, we evaluate the quantized model on CPU
        val_loader = get_dataloader(DEV_FILE, tokenizer, VAL_BATCH_SIZE, MAX_SEQ_LENGTH, LABEL2ID, shuffle=False)
        
        logger.info("Evaluating quantized model performance on Validation set (CPU)...")
        quantized_model = quantized_model.cpu()
        
        # Use our run_evaluation to evaluate the model
        token_report, entity_report, _, _, _, _ = run_evaluation(
            quantized_model, val_loader, tokenizer, ID2LABEL, None
        )
        logger.info(f"Quantized Model overall Accuracy on Val: {token_report['accuracy']:.4f}")
        logger.info(f"Quantized Model overall Entity F1-Score on Val: {entity_report['overall']['f1-score']:.4f}")

=======
import argparse
import os
import torch
from transformers import AutoTokenizer

from src.dataset import read_conll_data, extract_label_list, create_dataloader
from src.config import get_config
from src.model import get_model

try:
    from src.trainer import StandardTrainer
except ImportError:
    # Trainer provided by Dev 3, we mock if not found
    class StandardTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def train(self):
            print("Mock training...")
        def evaluate(self, loader):
            print("Mock evaluating...")

def parse_args():
    parser = argparse.ArgumentParser(description="NER Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "infer", "distill", "quantize"],
                        help="Pipeline running mode")
    parser.add_argument("--model_type", type=str, default="phobert", choices=["phobert", "phobert_lora", "lstm", "bilstm"],
                        help="Model type (implemented by Dev 2)")
    parser.add_argument("--trainer_type", type=str, default="standard", choices=["standard", "distillation", "quantization"],
                        help="Trainer type (implemented by Dev 3)")
    parser.add_argument("--data_dir", type=str, default="DATA",
                        help="Directory containing the dataset")
    parser.add_argument("--config_file", type=str, default="config.json",
                        help="Path to the config file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results and checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Starting run with mode: {args.mode}")
    
    # 1. Load Config
    config = get_config(args)
    print(f"Using model_name: {config.model_name}")
    
    # 2. Setup Data
    train_file = os.path.join(args.data_dir, "train_word.conll")
    dev_file = os.path.join(args.data_dir, "dev_word.conll")
    test_file = os.path.join(args.data_dir, "test_word.conll")
    
    # Load dataset
    print("Reading data...")
    train_texts, train_tags = read_conll_data(train_file)
    dev_texts, dev_tags = read_conll_data(dev_file)
    test_texts, test_tags = read_conll_data(test_file)
    
    label_list = extract_label_list(train_tags)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)
    
    print(f"Found {num_labels} labels: {label_list}")
    
    # Initialize Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # DataLoader
    print("Creating DataLoaders...")
    train_loader = create_dataloader(train_texts, train_tags, tokenizer, config.max_len, label2id, config.batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_texts, dev_tags, tokenizer, config.max_len, label2id, config.batch_size, shuffle=False)
    test_loader = create_dataloader(test_texts, test_tags, tokenizer, config.max_len, label2id, config.batch_size, shuffle=False)
    
    # 3. Initialize Model
    print("Initializing model...")
    model = get_model(config, num_labels)
    
    if model is None:
        print("Model is not fully implemented yet. Terminating program.")
        return
        
    # 4. Initialize Trainer
    print(f"Initializing Trainer of type: {args.trainer_type}")
    if args.trainer_type == "standard":
        try:
            trainer = StandardTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=dev_loader,
                config=config,
                id2label=id2label,
                device=config.device
            )
        except Exception as e:
            print(f"Error initializing Trainer: {e}")
            trainer = None
    else:
        print("Other trainers are not fully imported yet.")
        trainer = None
        
    # 5. Execute based on mode
    if args.mode == "train":
        if trainer:
            print("Starting training...")
            # trainer.train()
            print("Mock training completed.")
    elif args.mode == "evaluate":
        if trainer:
            print("Starting evaluation...")
            # trainer.evaluate(test_loader)
            print("Mock evaluation completed.")
    elif args.mode == "infer":
        print("Performing inference...")
    elif args.mode == "distill":
        print("Running Knowledge Distillation...")
    elif args.mode == "quantize":
        print("Running Quantization-Aware Training...")
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
