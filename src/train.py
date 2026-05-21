#!/usr/bin/env python3
"""
Optimized training script for Kaggle 2x T4 GPU
Maximizes both GPU utilization
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

import config
from src.data_loader import create_phobert_loaders, create_lstm_loaders, read_conll_file
from src.models import PhoBERTNER, PhoBERTLoRANER, LSTMCRF, StudentLSTM
from src.utils import set_seed, compute_metrics, plot_loss_curve, calculate_label_weights, get_error_analysis


def setup_multi_gpu(model, num_gpus):
    """
    Tối ưu DataParallel cho 2x GPU Kaggle
    """
    if num_gpus > 1:
        # Gán GPU 0 và 1 đều làm việc
        model = nn.DataParallel(model, device_ids=[0, 1])
        print(f"\n{'='*60}")
        print(f"✅ DataParallel ACTIVATED")
        print(f"   - Number of GPUs: {num_gpus}")
        print(f"   - Device IDs: [0, 1]")
        print(f"   - Name: nn.DataParallel")
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠️  Single GPU mode - DataParallel not needed")
    return model


def train_phobert_2gpu(train_loader, dev_loader, test_loader, num_labels, device, args):
    print(f"--- Bắt đầu huấn luyện: {args.model.upper()} (2x GPU Optimized) ---")
    writer = SummaryWriter(log_dir=os.path.join(config.RESULTS_DIR, f"logs/phobert_{args.model}"))
    
    num_gpus = torch.cuda.device_count()
    use_parallel = num_gpus > 1
    print(f"📊 GPU Info: {num_gpus} GPUs detected")
    
    if args.model == 'lora':
        model = PhoBERTLoRANER(config.PHOBERT_MODEL_NAME, num_labels).to(device)
        model.print_trainable_parameters()
    else:
        model = PhoBERTNER(config.PHOBERT_MODEL_NAME, num_labels, use_crf=False).to(device)
    
    if use_parallel:
        model = setup_multi_gpu(model, num_gpus)
        print(f"✅ Verification: Model wrapped with {model.__class__.__name__}")
        print(f"   Model device_ids: {model.device_ids if hasattr(model, 'device_ids') else 'N/A'}\n")
    else:
        print(f"⚠️  Single GPU mode detected\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    label_weights = calculate_label_weights(config.TRAIN_FILE, config.LABEL2ID).to(device)
    criterion = nn.CrossEntropyLoss(weight=label_weights, ignore_index=-100)
    
    scaler = GradScaler('cuda', enabled=args.amp)
    
    best_f1 = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    accumulation_steps = args.accumulation_steps

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                    valid_len = mask.sum().item()
                    all_preds.extend(p_seq[:valid_len].cpu().tolist())
                    all_labels.extend(l_seq[:valid_len].cpu().tolist())
                    
        val_loss /= len(dev_loader)
        val_losses.append(val_loss)
        metrics = compute_metrics(all_preds, all_labels, config.LABEL_LIST, ignore_index=-100)
        f1 = metrics['macro avg']['f1-score']
        
        print(f"--> Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val F1={f1:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metric/Val_F1', f1, epoch)
        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            model_state = model.module.state_dict() if use_parallel else model.state_dict()
            torch.save(model_state, os.path.join(config.SAVED_MODELS_DIR, f"phobert_{args.model}_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    print("--- Bắt đầu đánh giá trên tập Test ---")
    best_state = torch.load(os.path.join(config.SAVED_MODELS_DIR, f"phobert_{args.model}_best.pt"))
    if use_parallel:
        model.module.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    
    model.eval()
    test_preds, test_labels, test_preds_seq, test_labels_seq = [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(logits, dim=-1)
            for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                valid_len = mask.sum().item()
                valid_p = p_seq[:valid_len].cpu().tolist()
                valid_l = l_seq[:valid_len].cpu().tolist()
                test_preds.extend(valid_p)
                test_labels.extend(valid_l)
                test_preds_seq.append(valid_p)
                test_labels_seq.append(valid_l)
                
    test_metrics = compute_metrics(test_preds, test_labels, config.LABEL_LIST, ignore_index=-100)
    print(f"KẾT QUẢ TEST F1 (Macro): {test_metrics['macro avg']['f1-score']:.4f}")
    plot_loss_curve(train_losses, val_losses, os.path.join(config.PLOTS_DIR, f"phobert_{args.model}_loss.png"))
    writer.close()
    
    test_sentences, _ = read_conll_file(config.TEST_FILE)
    get_error_analysis(test_sentences, test_labels_seq, test_preds_seq, config.ID2LABEL, num_samples=5)


def train_lstm_2gpu(train_loader, dev_loader, test_loader, word2idx, num_labels, device, args):
    print("--- Bắt đầu huấn luyện LSTM (2x GPU Optimized) ---")
    writer = SummaryWriter(log_dir=os.path.join(config.RESULTS_DIR, "logs/lstm"))
    
    num_gpus = torch.cuda.device_count()
    use_parallel = num_gpus > 1
    print(f"📊 GPU Info: {num_gpus} GPUs detected")
    
    model = LSTMCRF(
        vocab_size=len(word2idx), 
        embedding_dim=config.LSTM_EMBEDDING_DIM, 
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_labels=num_labels, 
        dropout=config.LSTM_DROPOUT, 
        use_crf=config.LSTM_USE_CRF
    ).to(device)
    
    if use_parallel:
        model = setup_multi_gpu(model, num_gpus)
        print(f"✅ Verification: Model wrapped with {model.__class__.__name__}")
        print(f"   Model device_ids: {model.device_ids if hasattr(model, 'device_ids') else 'N/A'}\n")
    else:
        print(f"⚠️  Single GPU mode detected\n")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = GradScaler('cuda', enabled=args.amp)
    
    best_f1 = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    accumulation_steps = args.accumulation_steps

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss.mean()
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                    loss, emissions = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = loss.mean()
                    preds = model.module.decode(emissions, attention_mask=attention_mask) if use_parallel else model.decode(emissions, attention_mask=attention_mask)
                val_loss += loss.item()
                
                for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                    valid_len = mask.sum().item()
                    valid_p = p_seq[:valid_len] if isinstance(p_seq, list) else p_seq[:valid_len].cpu().tolist()
                    valid_l = l_seq[:valid_len].cpu().tolist()
                    all_preds.extend(valid_p)
                    all_labels.extend(valid_l)
                    
        val_loss /= len(dev_loader)
        val_losses.append(val_loss)
        metrics = compute_metrics(all_preds, all_labels, config.LABEL_LIST, ignore_index=0)
        f1 = metrics['macro avg']['f1-score']
        
        print(f"--> Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val F1={f1:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metric/Val_F1', f1, epoch)
        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            model_state = model.module.state_dict() if use_parallel else model.state_dict()
            torch.save(model_state, os.path.join(config.SAVED_MODELS_DIR, "lstm_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    print("--- Bắt đầu đánh giá trên tập Test ---")
    best_state = torch.load(os.path.join(config.SAVED_MODELS_DIR, "lstm_best.pt"))
    if use_parallel:
        model.module.load_state_dict(best_state)
    else:
        model.load_state_dict(best_state)
    
    model.eval()
    test_preds, test_labels, test_preds_seq, test_labels_seq = [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                _, emissions = model(input_ids, attention_mask=attention_mask, labels=labels)
                preds = model.module.decode(emissions, attention_mask=attention_mask) if use_parallel else model.decode(emissions, attention_mask=attention_mask)
            
            for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                valid_len = mask.sum().item()
                valid_p = p_seq[:valid_len] if isinstance(p_seq, list) else p_seq[:valid_len].cpu().tolist()
                valid_l = l_seq[:valid_len].cpu().tolist()
                test_preds.extend(valid_p)
                test_labels.extend(valid_l)
                test_preds_seq.append(valid_p)
                test_labels_seq.append(valid_l)
                
    test_metrics = compute_metrics(test_preds, test_labels, config.LABEL_LIST, ignore_index=0)
    print(f"KẾT QUẢ TEST F1 (Macro): {test_metrics['macro avg']['f1-score']:.4f}")
    plot_loss_curve(train_losses, val_losses, os.path.join(config.PLOTS_DIR, "lstm_loss.png"))
    writer.close()
    
    test_sentences, _ = read_conll_file(config.TEST_FILE)
    get_error_analysis(test_sentences, test_labels_seq, test_preds_seq, config.ID2LABEL, num_samples=5)


def train_kd_2gpu(train_loader, dev_loader, test_loader, tokenizer, num_labels, device, args):
    print(f"--- Knowledge Distillation (2x GPU Optimized) ---")
    writer = SummaryWriter(log_dir=os.path.join(config.RESULTS_DIR, "logs/kd_student"))
    
    num_gpus = torch.cuda.device_count()
    use_parallel = num_gpus > 1
    print(f"📊 GPU Info: {num_gpus} GPUs detected")
    
    teacher = PhoBERTNER(config.PHOBERT_MODEL_NAME, num_labels, use_crf=False).to(device)
    teacher_path = os.path.join(config.SAVED_MODELS_DIR, "phobert_phobert_best.pt")
    if not os.path.exists(teacher_path):
        print(f"❌ LỖI: Chưa tìm thấy Teacher tại {teacher_path}!")
        return
    
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    if use_parallel:
        teacher = setup_multi_gpu(teacher, num_gpus)
        print(f"✅ Teacher Model: {teacher.__class__.__name__} (device_ids={teacher.device_ids if hasattr(teacher, 'device_ids') else 'N/A'})")
    teacher.eval()
    
    student = StudentLSTM(
        vocab_size=tokenizer.vocab_size, 
        embedding_dim=config.LSTM_EMBEDDING_DIM, 
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_labels=num_labels,
        dropout=config.LSTM_DROPOUT
    ).to(device)
    if use_parallel:
        student = setup_multi_gpu(student, num_gpus)
        print(f"✅ Student Model: {student.__class__.__name__} (device_ids={student.device_ids if hasattr(student, 'device_ids') else 'N/A'})\n")
    else:
        print(f"⚠️  Single GPU mode detected\n")
    
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    label_weights = calculate_label_weights(config.TRAIN_FILE, config.LABEL2ID).to(device)
    criterion = nn.CrossEntropyLoss(weight=label_weights, ignore_index=-100)
    scaler = GradScaler('cuda', enabled=args.amp)
    
    temperature, alpha = args.kd_temp, args.kd_alpha
    best_f1, patience_counter = 0, 0
    train_losses, val_losses = [], []
    accumulation_steps = args.accumulation_steps

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train KD]")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                with torch.no_grad():
                    teacher_logits = teacher(input_ids, attention_mask=attention_mask)
                
                student_logits = student(input_ids, attention_mask=attention_mask)
                
                hard_loss = criterion(student_logits.view(-1, num_labels), labels.view(-1))
                soft_loss = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1)
                ) * (temperature ** 2)
                
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        student.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                    logits = student(input_ids, attention_mask=attention_mask)
                    loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                    valid_len = mask.sum().item()
                    all_preds.extend(p_seq[:valid_len].cpu().tolist())
                    all_labels.extend(l_seq[:valid_len].cpu().tolist())
                    
        val_loss /= len(dev_loader)
        val_losses.append(val_loss)
        metrics = compute_metrics(all_preds, all_labels, config.LABEL_LIST, ignore_index=-100)
        f1 = metrics['macro avg']['f1-score']
        
        print(f"--> Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val F1={f1:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metric/Val_F1', f1, epoch)
        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            model_state = student.module.state_dict() if use_parallel else student.state_dict()
            torch.save(model_state, os.path.join(config.SAVED_MODELS_DIR, "student_kd_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break

    print("--- Bắt đầu đánh giá trên tập Test ---")
    best_state = torch.load(os.path.join(config.SAVED_MODELS_DIR, "student_kd_best.pt"))
    if use_parallel:
        student.module.load_state_dict(best_state)
    else:
        student.load_state_dict(best_state)
    
    student.eval()
    test_preds, test_labels, test_preds_seq, test_labels_seq = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast('cuda', enabled=args.amp, dtype=torch.float16):
                logits = student(input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(logits, dim=-1)
            for p_seq, l_seq, mask in zip(preds, labels, attention_mask):
                valid_len = mask.sum().item()
                valid_p = p_seq[:valid_len].cpu().tolist()
                valid_l = l_seq[:valid_len].cpu().tolist()
                test_preds.extend(valid_p)
                test_labels.extend(valid_l)
                test_preds_seq.append(valid_p)
                test_labels_seq.append(valid_l)
                
    test_metrics = compute_metrics(test_preds, test_labels, config.LABEL_LIST, ignore_index=-100)
    print(f"KẾT QUẢ TEST F1 (Macro): {test_metrics['macro avg']['f1-score']:.4f}")
    plot_loss_curve(train_losses, val_losses, os.path.join(config.PLOTS_DIR, "kd_loss.png"))
    writer.close()
    
    test_sentences, _ = read_conll_file(config.TEST_FILE)
    get_error_analysis(test_sentences, test_labels_seq, test_preds_seq, config.ID2LABEL, num_samples=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['phobert', 'lstm', 'lora', 'kd'])
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument('--kd_temp', type=float, default=2.0)
    parser.add_argument('--kd_alpha', type=float, default=0.5)
    parser.add_argument('--amp', action='store_true', default=True, help='Enable mixed precision training')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"{'🚀 NER TRAINING - 2x GPU OPTIMIZED MODE':^70}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"")
    print(f"GPU Status:")
    print(f"  Total GPUs Available: {num_gpus} {'✅' if num_gpus >= 2 else '❌'}")
    
    # Check each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {gpu_name} - {gpu_memory:.1f}GB")
    
    print(f"")
    print(f"Training Settings:")
    print(f"  Mixed Precision (AMP): {'✅ Enabled' if args.amp else '❌ Disabled'}")
    print(f"  Gradient Accumulation: {args.accumulation_steps}x")
    print(f"  Multi-GPU Mode: {'✅ DataParallel (device_ids=[0, 1])' if num_gpus > 1 else '❌ Single GPU'}")
    print(f"{'='*70}\n")

    if args.model in ['phobert', 'lora', 'kd']:
        train_loader, dev_loader, test_loader, tokenizer = create_phobert_loaders(
            config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE,
            config.PHOBERT_MODEL_NAME, config.LABEL2ID, config.MAX_SEQ_LENGTH, args.batch_size
        )
        if args.model == 'kd':
            train_kd_2gpu(train_loader, dev_loader, test_loader, tokenizer, config.NUM_LABELS, device, args)
        else:
            train_phobert_2gpu(train_loader, dev_loader, test_loader, config.NUM_LABELS, device, args)
            
    elif args.model == 'lstm':
        train_loader, dev_loader, test_loader, word2idx = create_lstm_loaders(
            config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE,
            config.LABEL2ID, config.MAX_SEQ_LENGTH, args.batch_size, config.LSTM_MAX_VOCAB_SIZE
        )
        train_lstm_2gpu(train_loader, dev_loader, test_loader, word2idx, config.NUM_LABELS, device, args)


if __name__ == "__main__":
    main()
