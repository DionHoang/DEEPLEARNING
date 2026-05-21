# Các hàm tiện ích (metrics, visualization)
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed):
    """Cố định random seed cho tái lập kết quả"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels, label_list, ignore_index=-100):
    """
    Tính các chỉ số đánh giá: Precision, Recall, F1-score.
    preds, labels: list các label ids (flattened).
    Trả về classification_report dạng dict.
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
    true_label_names = [id2label.get(l, 'O') for l in filtered_labels]
    pred_label_names = [id2label.get(p, 'O') for p in filtered_preds]
    
    # Tạo report
    report = classification_report(
        true_label_names,
        pred_label_names,
        output_dict=True,
        zero_division=0
    )
    return report


def compute_entity_level_metrics(preds_sequences, labels_sequences, id2label):
    """
    Tính entity-level F1 (span-based evaluation).
    preds_sequences: list of list of label ids per sentence.
    labels_sequences: list of list of label ids per sentence.
    """
    def extract_entities(label_seq, id2label_map):
        entities = []
        entity = None
        for i, lid in enumerate(label_seq):
            tag = id2label_map.get(lid, 'O')
            if tag.startswith('B-'):
                if entity:
                    entities.append(entity)
                entity = {'type': tag[2:], 'start': i, 'end': i}
            elif tag.startswith('I-') and entity and tag[2:] == entity['type']:
                entity['end'] = i
            else:
                if entity:
                    entities.append(entity)
                    entity = None
        if entity:
            entities.append(entity)
        return set((e['type'], e['start'], e['end']) for e in entities)

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
                entity_type_stats[etype] = {'tp': 0, 'fp': 0, 'fn': 0}
            type_pred = {e for e in pred_entities if e[0] == etype}
            type_true = {e for e in true_entities if e[0] == etype}
            entity_type_stats[etype]['tp'] += len(type_pred & type_true)
            entity_type_stats[etype]['fp'] += len(type_pred - type_true)
            entity_type_stats[etype]['fn'] += len(type_true - type_pred)

    # Tính overall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-type
    per_type = {}
    for etype, stats in entity_type_stats.items():
        p = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        r = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_type[etype] = {'precision': p, 'recall': r, 'f1-score': f, 'support': stats['tp'] + stats['fn']}

    return {
        'overall': {'precision': precision, 'recall': recall, 'f1-score': f1},
        'per_type': per_type
    }


def plot_loss_curve(train_losses, val_losses, save_path):
    """Vẽ biểu đồ loss qua các epoch"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved loss plot to {save_path}")


def plot_f1_curve(train_f1s, val_f1s, save_path):
    """Vẽ biểu đồ F1-score qua các epoch"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label='Train F1', marker='o', linewidth=2)
    plt.plot(val_f1s, label='Validation F1', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Training & Validation F1-score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved F1 plot to {save_path}")


def plot_confusion_matrix(preds, labels, label_list, save_path, ignore_index=-100):
    """Vẽ confusion matrix cho các nhãn thực thể (bỏ O)"""
    id2label = {i: l for i, l in enumerate(label_list)}
    
    filtered = [(p, l) for p, l in zip(preds, labels) if l != ignore_index]
    if not filtered:
        return
    f_preds, f_labels = zip(*filtered)
    
    # Chỉ lấy các nhãn thực thể (bỏ O)
    entity_labels = [l for l in label_list if l != 'O']
    entity_ids = [label_list.index(l) for l in entity_labels]
    
    # Lọc chỉ lấy các mẫu có nhãn thực thể (true hoặc pred)
    mask = [(l in entity_ids or p in entity_ids) for p, l in zip(f_preds, f_labels)]
    if not any(mask):
        return
    
    entity_preds = [id2label.get(p, 'O') for p, m in zip(f_preds, mask) if m]
    entity_labels_filtered = [id2label.get(l, 'O') for l, m in zip(f_labels, mask) if m]
    
    present_labels = sorted(set(entity_preds) | set(entity_labels_filtered))
    
    cm = confusion_matrix(entity_labels_filtered, entity_preds, labels=present_labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_labels, yticklabels=present_labels)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix (Entity Labels)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved confusion matrix to {save_path}")


def plot_entity_distribution(labels, label_list, save_path, ignore_index=-100):
    """Vẽ biểu đồ phân bố nhãn thực thể"""
    id2label = {i: l for i, l in enumerate(label_list)}
    filtered = [id2label.get(l, 'O') for l in labels if l != ignore_index]
    entity_labels = [l for l in filtered if l != 'O']
    
    if not entity_labels:
        return
    
    from collections import Counter
    counter = Counter(entity_labels)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels_sorted, counts = zip(*sorted_items)
    
    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels_sorted)))
    plt.bar(labels_sorted, counts, color=colors)
    plt.xlabel('Entity Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Entity Label Distribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Saved entity distribution to {save_path}")


def save_metrics_csv(metrics_history, save_path):
    """Lưu metrics qua các epoch vào CSV"""
    import csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'f1_score'])
        for row in metrics_history:
            writer.writerow(row)
    print(f"  -> Saved metrics to {save_path}")


def save_predictions(sentences, true_labels, pred_labels, save_path, id2label):
    """Lưu kết quả dự đoán ra file text"""
    with open(save_path, 'w', encoding='utf-8') as f:
        for sent, trues, preds in zip(sentences, true_labels, pred_labels):
            for token, t, p in zip(sent, trues, preds):
                t_name = id2label.get(t, 'O') if isinstance(t, int) else t
                p_name = id2label.get(p, 'O') if isinstance(p, int) else p
                f.write(f"{token}\t{t_name}\t{p_name}\n")
            f.write("\n")
    print(f"  -> Saved predictions to {save_path}")


def count_parameters(model):
    """Đếm tổng số tham số và số tham số có thể huấn luyện"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_size(model, model_name="Model"):
    """In thông tin kích thước mô hình"""
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
    """Tính trọng số cho các nhãn dựa trên nghịch đảo tần suất xuất hiện."""
    counts = {l: 0 for l in label2id.keys()}
    with open(train_file, 'r', encoding='utf-8') as f:
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
    """So sánh nhãn thực tế và dự đoán để in ra các lỗi sai tiêu biểu."""
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
                sent_errors.append({
                    'word': sent[i] if i < len(sent) else "<PAD>",
                    'true': id2label.get(t, 'O'),
                    'pred': id2label.get(p, 'O')
                })
        
        if has_error:
            print(f"Câu: {' '.join(sent)}")
            for detail in sent_errors:
                print(f"  -> Từ: '{detail['word']}' | Nhãn gốc: {detail['true']} | Dự đoán: {detail['pred']}")
            print("-" * 30)
            errors_found += 1