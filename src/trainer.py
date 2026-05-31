import os
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return {"loss": total_loss / max(1, total_samples)}

    def on_train_start(self):
        pass

    def on_train_end(self, results):
        return results

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels, inputs)

        return loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        grad_clip: Optional[float] = None,
        early_stop: Optional[int] = None,
        log_interval: int = 50,
    ) -> Dict:
        self.on_train_start()

        best_val = None
        best_path = None
        no_improve = 0

        for epoch in range(1, epochs + 1):

            self.model.train()
            epoch_loss = 0.0
            total_samples = 0

            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
            )

            for step, batch in progress:

                self.optimizer.zero_grad()

                loss = self.training_step(batch)

                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        grad_clip,
                    )

                self.optimizer.step()

                batch_size = batch[1].size(0)

                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

                if step % log_interval == 0:

                    global_step = (epoch - 1) * len(train_loader) + step

                    self.writer.add_scalar(
                        "train/batch_loss",
                        loss.item(),
                        global_step,
                    )

                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_epoch_loss = epoch_loss / max(1, total_samples)

            self.writer.add_scalar(
                "train/epoch_loss",
                avg_epoch_loss,
                epoch,
            )

            self.logger.info(f"Epoch {epoch} train_loss=" f"{avg_epoch_loss:.4f}")

            val_metrics = None

            if val_loader is not None:

                val_metrics = self.evaluate(val_loader)

                val_loss = val_metrics.get(
                    "loss",
                    float("inf"),
                )

                self.writer.add_scalar(
                    "val/loss",
                    val_loss,
                    epoch,
                )

                self.logger.info(f"Epoch {epoch} val_loss={val_loss:.4f}")

            if self.scheduler is not None:

                try:

                    if isinstance(
                        self.scheduler,
                        optim.lr_scheduler.ReduceLROnPlateau,
                    ):

                        if val_metrics is not None:

                            self.scheduler.step(val_loss)

                        else:

                            self.logger.warning(
                                "ReduceLROnPlateau requires validation metrics "
                                "(val_loader is None). Scheduler not updated."
                            )

                    else:

                        self.scheduler.step()

                except Exception as e:

                    self.logger.warning(
                        "Scheduler.step() failed: %s",
                        e,
                    )

                current_lr = self.optimizer.param_groups[0]["lr"]

                self.writer.add_scalar(
                    "train/lr",
                    current_lr,
                    epoch,
                )

            extra = {"val_loss": (val_metrics.get("loss") if val_metrics else None)}

            path = self.save_checkpoint(
                epoch,
                name="model.pt",
                extra=extra,
            )

            if val_metrics is not None:

                current_val = val_metrics.get(
                    "loss",
                    float("inf"),
                )

            else:

                current_val = avg_epoch_loss

                self.logger.warning(
                    "Validation loader not provided. "
                    "Early stopping will monitor training loss."
                )

            if best_val is None or current_val < best_val:

                best_val = current_val
                best_path = path
                no_improve = 0

            else:

                no_improve += 1

            if early_stop and no_improve >= early_stop:
                self.logger.info("Early stopping triggered")
                break

        results = {
            "best_path": best_path,
            "best_val": best_val,
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
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        T = self.temperature
        p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
        p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
        kd_loss = self.kldiv(p_s, p_t) * (T * T)
        ce_loss = self.criterion(student_logits, labels)
        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

    def training_step(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        student_logits = self.model(inputs)

        return self._distillation_loss(
            student_logits,
            teacher_logits,
            labels,
        )


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
            self.logger.info(
                "Loading best checkpoint: %s",
                best_path,
            )

            self.load_checkpoint(best_path)

        self.logger.info("Converting QAT model to INT8...")

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
