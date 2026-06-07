import os
import shutil
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import glob
from .utils import setup_logger, print_model_size


class BaseTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[LRScheduler] = None,
        save_dir: str = "results/checkpoints",
        tensorboard_dir: str = "results/tensorboard",
        log_dir: str = "results/logs",
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=os.path.join(tensorboard_dir, run_name or "run")
        )
        self.logger = setup_logger(run_name or __name__, log_dir=log_dir)

        # log model size
        try:
            total, trainable = print_model_size(
                self.model, model_name=run_name or "Model"
            )
            self.logger.info(f"Model params: total={total:,}, trainable={trainable:,}")
        except Exception:
            # best-effort only; show exception details for debugging
            self.logger.exception("print_model_size failed")

    def save_checkpoint(
        self, epoch: int, name: str = "checkpoint.pt", extra: Optional[Dict] = None
    ):
        path = os.path.join(self.save_dir, f"{epoch:03d}_{name}")
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

        # Scan the list of checkpoints, skip files that contain "best"
        all_checkpoints = []
        for f in glob.glob(os.path.join(self.save_dir, f"*_{name}")):
            if "best" not in os.path.basename(f):
                all_checkpoints.append(f)

        # --- Helper: extract epoch number from filename for sorting ---
        def extract_epoch_number(filepath):
            filename = os.path.basename(filepath)
            try:
                # Slice the string to obtain the numeric epoch portion before the first underscore
                return int(filename.split("_")[0])
            except (ValueError, IndexError):
                return 0

        # Sort checkpoints by extracted epoch number
        all_checkpoints.sort(key=extract_epoch_number)

        # Remove older checkpoint files; wrap removal in try/except for safety
        if len(all_checkpoints) > 3:
            for old_ckpt in all_checkpoints[:-3]:
                try:
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)
                except Exception as e:
                    self.logger.warning(
                        "Failed to delete old checkpoint %s: %s", old_ckpt, e
                    )

        try:
            self.logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            self.logger.warning("Failed logging checkpoint save: %s", e)

        return path

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[torch.device] = None,
    ):
        ckpt = torch.load(
            path,
            map_location=map_location or self.device,
        )

        self.model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, RuntimeError) as e:
                self.logger.warning(
                    "Failed to load optimizer state: %s",
                    e,
                )

        if self.scheduler is not None and "scheduler_state" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except (ValueError, RuntimeError) as e:
                self.logger.warning(
                    "Failed to load scheduler state: %s",
                    e,
                )

        try:
            self.logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            self.logger.warning(
                "Failed logging checkpoint load: %s",
                e,
            )

        return ckpt

    def evaluate(self, dataloader: DataLoader) -> Dict:
        # Backwards-compatible wrapper that evaluates `self.model` on `self.device`.
        return self.evaluate_model(self.model, dataloader, device=self.device)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict:
        device = device or self.device
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if hasattr(model, "use_crf") and model.use_crf:
                    batch_preds = model.decode(inputs)
                    # Remove padding (-100) from labels to compare one-to-one with Viterbi results
                    for pred_seq, label_seq in zip(batch_preds, labels.cpu().tolist()):
                        valid_labels = [l for l in label_seq if l != -100]
                        for p, l in zip(pred_seq, valid_labels):
                            if p == l:
                                total_correct += 1
                            total_valid_tokens += 1
                else:
                    preds = torch.argmax(outputs, dim=-1)
                    mask = labels != -100
                    if preds.shape == labels.shape:
                        total_correct += ((preds == labels) & mask).sum().item()
                        total_valid_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = (
            total_correct / max(1, total_valid_tokens)
            if total_valid_tokens > 0
            else 0.0
        )

        return {"loss": avg_loss, "accuracy": accuracy}

    def on_train_start(self):
        pass

    def on_train_end(self, results):
        return results

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels, inputs)

        return loss, outputs, labels

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        grad_clip: Optional[float] = None,
        early_stop: Optional[int] = None,
        log_interval: int = 50,
        start_epoch: int = 1,
    ) -> Dict:
        self.on_train_start()

        best_val = None
        best_path = None
        no_improve = 0

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(start_epoch, epochs + 1):

            self.model.train()
            epoch_loss = 0.0
            total_samples = 0

            train_correct = 0
            train_valid_tokens = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}",
            )

            for step, batch in enumerate(progress):

                self.optimizer.zero_grad()

                loss, outputs, labels = self.training_step(batch)

                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        grad_clip,
                    )

                self.optimizer.step()

                if self.scheduler is not None and not isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

                batch_size = batch[1].size(0)

                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
                # Compute training accuracy
                if hasattr(self.model, "use_crf") and self.model.use_crf:
                    inputs = batch[0].to(self.device)

                    batch_preds = self.model.decode(inputs)
                    for pred_seq, label_seq in zip(batch_preds, labels.cpu().tolist()):
                        valid_labels = [l for l in label_seq if l != -100]
                        for p, l in zip(pred_seq, valid_labels):
                            if p == l:
                                train_correct += 1
                            train_valid_tokens += 1
                else:
                    preds = torch.argmax(outputs, dim=-1)
                    mask = labels != -100
                    if preds.shape == labels.shape:
                        train_correct += ((preds == labels) & mask).sum().item()
                        train_valid_tokens += mask.sum().item()

                if step % log_interval == 0:

                    global_step = (epoch - 1) * len(train_loader) + step

                    self.writer.add_scalar(
                        "train/batch_loss",
                        loss.item(),
                        global_step,
                    )

                progress.set_postfix(loss=f"{loss.item():.4f}")

            progress.close()

            avg_epoch_loss = epoch_loss / max(1, total_samples)
            train_acc = (
                train_correct / max(1, train_valid_tokens)
                if train_valid_tokens > 0
                else 0.0
            )

            history["train_loss"].append(avg_epoch_loss)
            history["train_acc"].append(train_acc)

            self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
            self.writer.add_scalar("train/epoch_acc", train_acc, epoch)
            self.logger.info(
                f"Epoch {epoch} train_loss={avg_epoch_loss:.4f}, train_acc={train_acc:.4f}"
            )

            # Evaluate on validation set
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss", float("inf"))
                val_acc = val_metrics.get("accuracy", 0.0)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/acc", val_acc, epoch)
                self.logger.info(
                    f"Epoch {epoch} val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

            # Learning rate scheduler step handling
            if self.scheduler is not None:
                try:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics is not None:
                            self.scheduler.step(val_loss)
                        else:
                            self.logger.warning(
                                "ReduceLROnPlateau requires val_loader. Scheduler skipped."
                            )
                except Exception as e:
                    self.logger.warning(f"Scheduler.step() failed: {e}")

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", current_lr, epoch)

            extra = {"val_loss": val_metrics.get("loss") if val_metrics else None}
            path = self.save_checkpoint(epoch, name="model.pt", extra=extra)

            current_val = (
                val_metrics.get("loss", float("inf")) if val_metrics else avg_epoch_loss
            )

            if best_val is None or current_val < best_val:
                best_val = current_val
                no_improve = 0
                best_path = os.path.join(self.save_dir, "best_model.pt")
                try:
                    shutil.copy(path, best_path)
                    self.logger.info(
                        f"New best model saved at epoch {epoch} with val_loss {best_val:.4f}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to copy best model: {e}")
                    best_path = path
            else:
                no_improve += 1

            if early_stop and no_improve >= early_stop:
                self.logger.info("Early stopping triggered")
                break

        results = {
            "best_path": best_path,
            "best_val": best_val,
            "history": history,
        }
        results = self.on_train_end(results)
        self.writer.close()
        return results


class DistillationTrainer(BaseTrainer):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        alpha: float = 0.5,
        temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(student_model, optimizer, criterion, device, **kwargs)
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.kldiv = nn.KLDivLoss(reduction="none")

    def _distillation_loss(self, student_logits, teacher_logits, labels, input_ids):
        T = self.temperature
        p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
        p_t = nn.functional.softmax(teacher_logits / T, dim=-1)

        # Use attention_mask (derived from input_ids) instead of labels != -100
        pad_token_id = 1
        attention_mask = (input_ids != pad_token_id).float()

        # Compute KL divergence per token
        kd_loss_per_token = self.kldiv(p_s, p_t).sum(dim=-1) * (T * T)

        # Apply attention_mask and average over valid tokens
        kd_loss = (kd_loss_per_token * attention_mask).sum() / torch.clamp(
            attention_mask.sum(), min=1e-8
        )

        ce_loss = self.criterion(student_logits, labels)

        # Normalize CE loss when the student model uses a CRF layer.
        if hasattr(self.model, "use_crf") and self.model.use_crf:
            num_valid_tokens = (labels != -100).sum().float()
            ce_loss = ce_loss / torch.clamp(num_valid_tokens, min=1e-8)

        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        student_logits = self.model(inputs)

        # Pass inputs into the loss function to compute an attention mask when needed
        loss = self._distillation_loss(student_logits, teacher_logits, labels, inputs)

        return loss, student_logits, labels


class QuantizationTrainer(BaseTrainer):
    """
    Trainer wrapper for Quantization Aware Training (QAT).
    """

    def __init__(
        self,
        *args,
        quantize_utils=None,
        force_convert_to_cpu: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.quantize_utils = quantize_utils
        self.force_convert_to_cpu = force_convert_to_cpu
        self.quantized_model = None

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.warning(
                "Failed to move model to device %s: %s",
                self.device,
                e,
            )

    def prepare_qat(self):
        if self.quantize_utils is None:
            raise RuntimeError("quantize_utils not provided")

        self.model = self.quantize_utils.prepare_qat(self.model)

        self.logger.info("Model prepared for QAT " "(FakeQuant nodes attached)")

        return self.model

    def convert_qat(self):
        if self.quantize_utils is None:
            raise RuntimeError("quantize_utils not provided")

        model_for_convert = self.model

        if self.force_convert_to_cpu:
            try:
                model_for_convert = model_for_convert.to(torch.device("cpu"))
            except Exception as e:
                self.logger.warning(
                    "Failed to move model to CPU " "for conversion: %s",
                    e,
                )

        qmodel = self.quantize_utils.convert_qat(model_for_convert)

        self.logger.info("Model converted to INT8")

        return qmodel

    def on_train_start(self):
        self.logger.info("=== STARTING QAT PIPELINE ===")

        if not hasattr(self.model, "qconfig") or self.model.qconfig is None:
            self.logger.info("Auto preparing model for QAT...")

            self.prepare_qat()

    def on_train_end(self, results):

        best_path = results.get("best_path")

        if best_path is not None and os.path.exists(best_path):
            self.logger.info("Loading best checkpoint: %s", best_path)
            self.load_checkpoint(best_path)
        else:
            self.logger.warning(
                "No best checkpoint found! Converting the final epoch state to INT8. This may degrade performance."
            )

        quantized_model = self.convert_qat()

        self.quantized_model = quantized_model

        results["quantized_model"] = quantized_model

        return results

    def evaluate_quantized(
        self,
        dataloader: DataLoader,
    ):
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. " "Train first.")

        try:
            params = list(self.quantized_model.parameters())

            device = params[0].device if len(params) > 0 else torch.device("cpu")

        except Exception:
            device = torch.device("cpu")

        return self.evaluate_model(
            self.quantized_model,
            dataloader,
            device=device,
        )
