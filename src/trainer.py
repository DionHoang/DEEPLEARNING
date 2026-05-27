import os
import time
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from .utils import setup_logger, print_model_size


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_dir: str = "results/checkpoints",
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=os.path.join("results", "tensorboard", run_name or "run")
        )
        # logger
        self.logger = setup_logger(run_name or __name__)
        # log model size
        try:
            total, trainable = print_model_size(
                self.model, model_name=run_name or "Model"
            )
            self.logger.info(f"Model params: total={total:,}, trainable={trainable:,}")
        except Exception:
            # best-effort only; show exception details for debugging
            self.logger.exception("print_model_size failed")
        if _HAS_WANDB and run_name:
            wandb.init(project="ner_project", name=run_name)

    def save_checkpoint(
        self, epoch: int, name: str = "checkpoint.pt", extra: Optional[Dict] = None
    ):
        path = os.path.join(self.save_dir, f"{epoch:03d}_{name}")
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        try:
            self.logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            self.logger.warning("Failed logging checkpoint save: %s", e)
        return path

    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, RuntimeError) as e:
                self.logger.warning("Failed to load optimizer state: %s", e)
        try:
            self.logger.info(f"Loaded checkpoint: {path}")
        except Exception as e:
            self.logger.warning("Failed logging checkpoint load: %s", e)
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
        """Evaluate an arbitrary model on a dataloader using the given device.

        This avoids swapping `self.model` in-place when evaluating transformed/quantized copies.
        """
        device = device or self.device
        model = model.to(device)
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                n_batches += 1
        return {"loss": total_loss / max(1, n_batches)}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        grad_clip: Optional[float] = None,
        early_stop: Optional[int] = None,
        log_interval: int = 50,
    ):
        self.logger.info(f"Starting training: epochs={epochs}")
        best_val = float("inf")
        best_path = None
        no_improve = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            progress = tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
            )
            for step, batch in progress:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                if self.scheduler:
                    try:
                        self.scheduler.step()
                    except Exception as e:
                        self.logger.warning("Scheduler.step() failed: %s", e)
                epoch_loss += loss.item()
                if step % log_interval == 0:
                    self.writer.add_scalar(
                        "train/batch_loss",
                        loss.item(),
                        epoch * len(train_loader) + step,
                    )
                    if _HAS_WANDB:
                        wandb.log({"train/batch_loss": loss.item()})
                progress.set_postfix(loss=loss.item())

            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
            try:
                self.logger.info(f"Epoch {epoch} train_loss={avg_epoch_loss:.4f}")
            except Exception as e:
                self.logger.warning("Logging epoch train loss failed: %s", e)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.writer.add_scalar("val/loss", val_metrics.get("loss", 0.0), epoch)
                try:
                    self.logger.info(
                        f"Epoch {epoch} val_loss={val_metrics.get('loss', 0.0):.4f}"
                    )
                except Exception as e:
                    self.logger.warning("Logging epoch val loss failed: %s", e)
                if _HAS_WANDB:
                    wandb.log({"val/loss": val_metrics.get("loss", 0.0)})

            # checkpointing
            extra = {"val_loss": val_metrics.get("loss") if val_metrics else None}
            path = self.save_checkpoint(epoch, name="model.pt", extra=extra)
            if val_metrics and val_metrics.get("loss", float("inf")) < best_val:
                best_val = val_metrics.get("loss")
                best_path = path
                no_improve = 0
            else:
                no_improve += 1

            if early_stop and no_improve >= early_stop:
                self.logger.info("Early stopping triggered")
                break

        self.logger.info(f"Training finished. best_val={best_val}")
        return {"best_path": best_path, "best_val": best_val}


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
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        T = self.temperature
        p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
        p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
        kd_loss = self.kldiv(p_s, p_t) * (T * T)
        ce_loss = self.criterion(student_logits, labels)
        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        **kwargs,
    ):
        best = {"best_val": float("inf"), "best_path": None}
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"KD Epoch {epoch}"):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                self.optimizer.zero_grad()
                student_logits = self.model(inputs)
                loss = self._distillation_loss(student_logits, teacher_logits, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            val_metrics = None
            if val_loader:
                val_metrics = self.evaluate(val_loader)
            path = self.save_checkpoint(
                epoch,
                name="student_distilled.pt",
                extra={"val_loss": val_metrics.get("loss") if val_metrics else None},
            )
            try:
                self.logger.info(
                    f"KD Epoch {epoch} loss={epoch_loss / max(1, len(train_loader)):.4f}"
                )
            except Exception as e:
                self.logger.warning("Logging KD epoch loss failed: %s", e)
            if val_metrics and val_metrics.get("loss", float("inf")) < best["best_val"]:
                best["best_val"] = val_metrics.get("loss")
                best["best_path"] = path

        return best


class QuantizationTrainer(BaseTrainer):
    """Trainer wrapper to support Quantization-Aware Training (QAT) flows.

    This class orchestrates preparing the model for QAT, running training on CPU,
    evaluating with fake quantization, and converting to a real INT8 quantized model.
    """

    def __init__(
        self, *args, quantize_utils=None, force_convert_to_cpu: bool = True, **kwargs
    ):
        """QuantizationTrainer accepts the same args as BaseTrainer.

        - `force_convert_to_cpu` (default True): keep QAT training on the provided device
          (GPU if available), but move the model to CPU only when performing final convert.
        """
        # Do not force CPU globally; allow QAT to run on GPU if user requested that.
        super().__init__(*args, **kwargs)
        self.quantize_utils = quantize_utils
        self.force_convert_to_cpu = force_convert_to_cpu
        # Ensure model is on the configured training device
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.warning("Failed to move model to device %s: %s", self.device, e)

    def prepare_qat(self):
        if not self.quantize_utils:
            raise RuntimeError("quantize_utils not provided")
        # Gọi helper của bạn để cấu hình qconfig và chèn FakeQuantize nodes
        self.model = self.quantize_utils.prepare_qat(self.model)
        try:
            self.logger.info("Model prepared for QAT (FakeQuant nodes attached)")
        except Exception as e:
            self.logger.warning("Logging prepare_qat failed: %s", e)
        return self.model

    def convert_qat(self):
        if not self.quantize_utils:
            raise RuntimeError("quantize_utils not provided")
        # Move to CPU for convert if requested (conversion often expects CPU-backed kernels)
        model_for_convert = self.model
        if self.force_convert_to_cpu:
            try:
                model_for_convert = self.model.to(torch.device("cpu"))
            except Exception as e:
                self.logger.warning("Failed to move model to CPU for convert: %s", e)

        # Gọi helper của bạn để biến đổi mô hình sang cấu trúc INT8 thực tế
        qmodel = self.quantize_utils.convert_qat(model_for_convert)
        try:
            self.logger.info("Model converted from QAT to real INT8 quantized version")
        except Exception as e:
            self.logger.warning("Logging convert_qat failed: %s", e)
        return qmodel

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        grad_clip: Optional[float] = None,
        early_stop: Optional[int] = None,
        log_interval: int = 50,
    ):
        """Thực hiện toàn bộ quy trình QAT khép kín."""
        self.logger.info("=== BẮT ĐẦU QUY TRÌNH QUANTIZATION AWARE TRAINING (QAT) ===")

        # Bước 1: Tự động chuyển đổi mô hình sang trạng thái QAT (nếu chưa gọi ngoài)
        # Kiểm tra xem mô hình đã có thuộc tính qconfig chưa, nếu chưa thì tự kích hoạt
        if not hasattr(self.model, "qconfig") or self.model.qconfig is None:
            self.logger.info("Auto-preparing model for QAT...")
            self.prepare_qat()

        # Bước 2: Kế thừa hàm train gốc của BaseTrainer để fine-tune mô hình với lỗi lượng tử hóa giả lập
        # Hàm này sẽ tự động handle TensorBoard, Logging, Early stopping, Checkpoint
        results = super().train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            grad_clip=grad_clip,
            early_stop=early_stop,
            log_interval=log_interval,
        )

        # Bước 3: Sau khi train xong, load lại checkpoint tốt nhất (best_path) để convert
        if results.get("best_path") and os.path.exists(results["best_path"]):
            self.logger.info(
                f"Loading best checkpoint for final conversion: {results['best_path']}"
            )
            self.load_checkpoint(results["best_path"])

        # Bước 4: Convert sang mô hình INT8 thực tế
        self.logger.info("Converting fine-tuned QAT model to physical INT8 model...")
        quantized_model = self.convert_qat()

        # Bước 5: Đánh giá độ chính xác cuối cùng của mô hình INT8 thật
        if val_loader is not None:
            # Evaluate the quantized model without swapping `self.model` in-place.
            # Determine device of quantized_model (most likely CPU if conversion forced).
            try:
                params = list(quantized_model.parameters())
                q_device = params[0].device if params else torch.device("cpu")
            except Exception as e:
                self.logger.warning(
                    "Failed to infer quantized model device, falling back to CPU: %s", e
                )
                q_device = torch.device("cpu")

            final_metrics = self.evaluate_model(
                quantized_model, val_loader, device=q_device
            )

            try:
                self.logger.info("--- KẾT QUẢ CUỐI CÙNG ---")
                self.logger.info(
                    f"Độ chính xác / Loss của mô hình INT8 trên tập Val: {final_metrics}"
                )
            except Exception as e:
                self.logger.warning("Logging final quantized metrics failed: %s", e)

        # Trả về mô hình đã nén để người dùng đem đi deploy (.pt / torchscript)
        return quantized_model
