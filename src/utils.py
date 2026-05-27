import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

import logging
from pathlib import Path
from datetime import datetime


def set_seed(seed):
    """
    Set random seeds for reproducibility across libraries and CUDA.

    This function sets the seed for Python's `random`, NumPy, and PyTorch
    (including CUDA). It also configures cuDNN for deterministic behavior.

    Parameters
    ---
    seed : int
        The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels, label_list, ignore_index=-100):
    """
    Compute token-level classification metrics and return a sklearn-like report.

    This function filters out positions with `ignore_index` (e.g., -100 used for
    padding), maps label ids to label names using `label_list`, and returns the
    result of `sklearn.metrics.classification_report` as a dictionary.

    Parameters
    ---
    preds : Sequence[int]
        Flattened list of predicted label ids.
    labels : Sequence[int]
        Flattened list of ground-truth label ids.
    label_list : Sequence[str]
        List mapping label id -> label name.
    ignore_index : int, optional
        Label id to ignore (default: -100).

    Returns
    ---
    dict
        Classification report as returned by `classification_report(output_dict=True)`.
    """
    id2label = {i: l for i, l in enumerate(label_list)}

    # Lọc bỏ các vị trí ignore
    filtered_preds = []
    filtered_labels = []
    for p, l in zip(preds, labels):
        if l != ignore_index:
            filtered_preds.append(p)
            filtered_labels.append(l)

    # Chuyển đổi sang tên nhãn
    true_label_names = [id2label.get(l, "O") for l in filtered_labels]
    pred_label_names = [id2label.get(p, "O") for p in filtered_preds]

    # Tạo report
    report = classification_report(
        true_label_names, pred_label_names, output_dict=True, zero_division=0
    )
    return report


def compute_entity_level_metrics(preds_sequences, labels_sequences, id2label):
    """
    Compute span/entity-level precision, recall and F1 (span-based evaluation).

    The evaluation extracts BIO-style spans from sequences of label ids and
    computes true positives, false positives and false negatives both overall
    and per entity type.

    Parameters
    ---
    preds_sequences : Sequence[Sequence[int]]
        Predicted label id sequences (one sequence per example).
    labels_sequences : Sequence[Sequence[int]]
        Ground-truth label id sequences.
    id2label : dict
        Mapping from label id to label name (e.g., {0: 'O', 1: 'B-PER', ...}).

    Returns
    ---
    dict
        A dictionary with keys `'overall'` (precision/recall/f1) and `'per_type'`
        containing per-entity-type metrics and support counts.
    """

    def extract_entities(label_seq, id2label_map):
        entities = []
        entity = None
        for i, lid in enumerate(label_seq):
            tag = id2label_map.get(lid, "O")
            if tag.startswith("B-"):
                if entity:
                    entities.append(entity)
                entity = {"type": tag[2:], "start": i, "end": i}
            elif tag.startswith("I-") and entity and tag[2:] == entity["type"]:
                entity["end"] = i
            else:
                if entity:
                    entities.append(entity)
                    entity = None
        if entity:
            entities.append(entity)
        return set((e["type"], e["start"], e["end"]) for e in entities)

    total_tp, total_fp, total_fn = 0, 0, 0
    entity_type_stats = {}

    for pred_seq, label_seq in zip(preds_sequences, labels_sequences):
        pred_entities = extract_entities(pred_seq, id2label)
        true_entities = extract_entities(label_seq, id2label)

        tp = pred_entities & true_entities
        fp = pred_entities - true_entities
        fn = true_entities - pred_entities

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        # Per-type stats
        all_types = set(e[0] for e in pred_entities | true_entities)
        for etype in all_types:
            if etype not in entity_type_stats:
                entity_type_stats[etype] = {"tp": 0, "fp": 0, "fn": 0}
            type_pred = {e for e in pred_entities if e[0] == etype}
            type_true = {e for e in true_entities if e[0] == etype}
            entity_type_stats[etype]["tp"] += len(type_pred & type_true)
            entity_type_stats[etype]["fp"] += len(type_pred - type_true)
            entity_type_stats[etype]["fn"] += len(type_true - type_pred)

    # Tính overall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Per-type
    per_type = {}
    for etype, stats in entity_type_stats.items():
        p = (
            stats["tp"] / (stats["tp"] + stats["fp"])
            if (stats["tp"] + stats["fp"]) > 0
            else 0
        )
        r = (
            stats["tp"] / (stats["tp"] + stats["fn"])
            if (stats["tp"] + stats["fn"]) > 0
            else 0
        )
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_type[etype] = {
            "precision": p,
            "recall": r,
            "f1-score": f,
            "support": stats["tp"] + stats["fn"],
        }

    return {
        "overall": {"precision": precision, "recall": recall, "f1-score": f1},
        "per_type": per_type,
    }


def plot_loss_curve(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves and save to file.

    Parameters
    ---
    train_losses : Sequence[float]
        Training loss values per epoch.
    val_losses : Sequence[float]
        Validation loss values per epoch.
    save_path : str
        Path where the PNG plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved loss plot to {save_path}")


def plot_f1_curve(train_f1s, val_f1s, save_path):
    """
    Plot training and validation F1-score curves and save to file.

    Parameters
    ---
    train_f1s : Sequence[float]
        Training F1 values per epoch.
    val_f1s : Sequence[float]
        Validation F1 values per epoch.
    save_path : str
        Path where the PNG plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label="Train F1", marker="o", linewidth=2)
    plt.plot(val_f1s, label="Validation F1", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    plt.title("Training & Validation F1-score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved F1 plot to {save_path}")


def plot_confusion_matrix(preds, labels, label_list, save_path, ignore_index=-100):
    """
    Plot a confusion matrix restricted to entity labels (excludes 'O') and save it.

    Parameters
    ---
    preds : Sequence[int]
        Flattened predicted label ids.
    labels : Sequence[int]
        Flattened true label ids.
    label_list : Sequence[str]
        Mapping from id -> label name.
    save_path : str
        Path where the PNG plot will be saved.
    ignore_index : int, optional
        Label id to ignore (default: -100).
    """
    id2label = {i: l for i, l in enumerate(label_list)}

    filtered = [(p, l) for p, l in zip(preds, labels) if l != ignore_index]
    if not filtered:
        return
    f_preds, f_labels = zip(*filtered)

    # Chỉ lấy các nhãn thực thể (bỏ O)
    entity_labels = [l for l in label_list if l != "O"]
    entity_ids = [label_list.index(l) for l in entity_labels]

    # Lọc chỉ lấy các mẫu có nhãn thực thể (true hoặc pred)
    mask = [(l in entity_ids or p in entity_ids) for p, l in zip(f_preds, f_labels)]
    if not any(mask):
        return

    entity_preds = [id2label.get(p, "O") for p, m in zip(f_preds, mask) if m]
    entity_labels_filtered = [id2label.get(l, "O") for l, m in zip(f_labels, mask) if m]

    present_labels = sorted(set(entity_preds) | set(entity_labels_filtered))

    cm = confusion_matrix(entity_labels_filtered, entity_preds, labels=present_labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix (Entity Labels)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved confusion matrix to {save_path}")


def plot_entity_distribution(labels, label_list, save_path, ignore_index=-100):
    """
    Plot the distribution (counts) of entity labels found in `labels`.

    Parameters
    ---
    labels : Sequence[int]
        Flattened label ids (e.g., across dataset).
    label_list : Sequence[str]
        Mapping from id -> label name.
    save_path : str
        Path where the PNG plot will be saved.
    ignore_index : int, optional
        Label id to ignore (default: -100).
    """
    id2label = {i: l for i, l in enumerate(label_list)}
    filtered = [id2label.get(l, "O") for l in labels if l != ignore_index]
    entity_labels = [l for l in filtered if l != "O"]

    if not entity_labels:
        return

    from collections import Counter

    counter = Counter(entity_labels)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels_sorted, counts = zip(*sorted_items)

    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_sorted)))
    plt.bar(labels_sorted, counts, color=colors)
    plt.xlabel("Entity Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Entity Label Distribution", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved entity distribution to {save_path}")


def save_metrics_csv(metrics_history, save_path):
    """
    Save training metrics history to a CSV file.

    Parameters
    ---
    metrics_history : Sequence[Sequence]
        Iterable of rows (e.g., epoch, train_loss, val_loss, precision, recall, f1_score).
    save_path : str
        Path to the output CSV file.
    """
    import csv

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "precision", "recall", "f1_score"]
        )
        for row in metrics_history:
            writer.writerow(row)
    print(f"  -> Saved metrics to {save_path}")


def save_predictions(sentences, true_labels, pred_labels, save_path, id2label):
    """
    Save token-level predictions to a TSV-style text file.

    Each line contains: token \t true_label \t pred_label. Sentences are separated by a blank line.

    Parameters
    ---
    sentences : Sequence[Sequence[str]]
        Tokenized sentences.
    true_labels : Sequence[Sequence[int]]
        Ground-truth label ids per sentence.
    pred_labels : Sequence[Sequence[int]]
        Predicted label ids per sentence.
    save_path : str
        Path to the output text file.
    id2label : dict
        Mapping from label id to label name.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for sent, trues, preds in zip(sentences, true_labels, pred_labels):
            for token, t, p in zip(sent, trues, preds):
                t_name = id2label.get(t, "O") if isinstance(t, int) else t
                p_name = id2label.get(p, "O") if isinstance(p, int) else p
                f.write(f"{token}\t{t_name}\t{p_name}\n")
            f.write("\n")
    print(f"  -> Saved predictions to {save_path}")


def count_parameters(model):
    """
    Count total and trainable parameters in a PyTorch model.

    Parameters
    ---
    model : torch.nn.Module
        The model to inspect.

    Returns
    ---
    tuple
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_size(model, model_name="Model"):
    """
    Print a short summary of model parameter counts and trainable ratio.

    Parameters
    ---
    model : torch.nn.Module
        The model to inspect.
    model_name : str, optional
        Friendly name used in the printed header.
    """
    total, trainable = count_parameters(model)
    print(f"\n{'='*50}")
    print(f" {model_name} Parameters")
    print(f"{'='*50}")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {total - trainable:>12,}")
    print(f"  Trainable ratio:      {trainable/total*100:>11.2f}%")
    print(f"{'='*50}\n")
    return total, trainable


def calculate_label_weights(train_file, label2id):
    """
    Compute class weights from a training file using inverse frequency.

    The function expects a token-per-line CoNLL-like format where the last column
    is the label. It counts occurrences per label then computes weights as
    `total / (count + eps)` and normalizes by the mean.

    Parameters
    ---
    train_file : str
        Path to the training file (CoNLL-like: token ... label per line).
    label2id : dict
        Mapping from label name to label id.

    Returns
    ---
    torch.Tensor
        A 1-D tensor of normalized class weights ordered by label id.
    """
    counts = {l: 0 for l in label2id.keys()}
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                tag = parts[-1]
                if tag in counts:
                    counts[tag] += 1

    total = sum(counts.values())
    weights = []
    for label in sorted(label2id, key=label2id.get):
        count = counts[label]
        # Thêm 1e-5 để tránh chia cho 0
        w = total / (count + 1e-5)
        weights.append(w)

    # Chuẩn hóa trọng số
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.mean()
    return weights


def get_error_analysis(sentences, true_labels, pred_labels, id2label, num_samples=5):
    """
    Print a small sample of error cases for human inspection.

    The function iterates over sentence-level predictions and prints up to
    `num_samples` sentences that contain at least one incorrect token label
    (ignoring padding labels such as -100).

    Parameters
    ---
    sentences : Sequence[Sequence[str]]
        Tokenized sentences.
    true_labels : Sequence[Sequence[int]]
        Ground-truth label ids.
    pred_labels : Sequence[Sequence[int]]
        Predicted label ids.
    id2label : dict
        Mapping from label id to label name.
    num_samples : int, optional
        Maximum number of error sentences to print (default: 5).
    """
    print(f"\n{'='*50}")
    print(" PHÂN TÍCH LỖI (ERROR ANALYSIS)")
    print(f"{'='*50}")
    errors_found = 0

    for sent, trues, preds in zip(sentences, true_labels, pred_labels):
        if errors_found >= num_samples:
            break

        has_error = False
        sent_errors = []
        for i, (t, p) in enumerate(zip(trues, preds)):
            # Bỏ qua các nhãn padding (-100) và nhãn đúng
            if t != p and t != -100:
                has_error = True
                sent_errors.append(
                    {
                        "word": sent[i] if i < len(sent) else "<PAD>",
                        "true": id2label.get(t, "O"),
                        "pred": id2label.get(p, "O"),
                    }
                )

        if has_error:
            print(f"Câu: {' '.join(sent)}")
            for detail in sent_errors:
                print(
                    f"  -> Từ: '{detail['word']}' | Nhãn gốc: {detail['true']} | Dự đoán: {detail['pred']}"
                )
            print("-" * 30)
            errors_found += 1


def setup_logger(name, log_dir="logs", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance with both console and file handlers.

    This function ensures that a log directory exists and generates a log file
    named with the current date if no specific file is provided.

    Parameters
    ---
    name : str
            The name of the logger (usually __name__).
    log_dir : str, optional
            Directory to save log files. Defaults to "logs".
    log_file : str, optional
            Specific path for the log file. If None, a default name based on the date is generated.
    level : int, optional
            The logging level (e.g., logging.INFO). Defaults to logging.INFO.

    Returns
    ---
    logging.Logger
            A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Respect user-provided `log_dir`. Only use package default when not provided or empty.
    if log_dir is None or str(log_dir).strip() == "":
        log_dir_path = Path(__file__).parent.parent / "logs"
    else:
        log_dir_path = Path(log_dir)

    # File handler
    if log_file is None:
        try:
            if not log_dir_path.exists():
                os.makedirs(log_dir_path, exist_ok=True)
        except OSError:
            # Fallback to current working directory if creating the dir fails
            log_dir_path = Path(os.getcwd()) / "logs"
            os.makedirs(log_dir_path, exist_ok=True)

        # File name: logs/training_YYYY-MM-DD.log
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir_path, f"training_{current_date}.log")
    else:
        # If user provides specific path, ensure directory exists
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except (TypeError, OSError):
            # If path creation fails, let FileHandler raise later with a clear error
            pass

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Disable propagate to prevent logs from being pushed to root logger (avoid duplicate printing if root logger also has handler)
    logger.propagate = False

    return logger
