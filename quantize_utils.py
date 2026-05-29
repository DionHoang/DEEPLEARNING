"""Quantization helpers: PTQ and QAT helpers for PyTorch models.

Utilities provided here are deliberately small and focused:
- `quantize_dynamic_ptq` for simple dynamic quantization (good for LSTM/Linear-heavy models)
- `prepare_qat` to attach QAT qconfigs to a model
- `convert_qat` to convert a QAT-prepared model to a quantized version

Note: For transformer-based models (PhoBERT), production-ready 4-bit/8-bit quantization
often requires specialized libraries (bitsandbytes / huggingface accelerate). These helpers
provide a baseline path using native torch.quantization where applicable.
"""

import torch
import torch.nn as nn
from .utils import setup_logger, print_model_size

logger = setup_logger(__name__)


def quantize_dynamic_ptq(
    model: nn.Module, dtype=torch.qint8, operators_to_quantize=None
) -> nn.Module:
    """Apply PyTorch dynamic quantization to model.

    Args:
            model: nn.Module to quantize
            dtype: target dtype, usually torch.qint8
            operators_to_quantize: iterable of types to quantize (defaults to Linear, LSTM)
    Returns:
            Quantized model (a new model instance)
    """
    if operators_to_quantize is None:
        operators_to_quantize = {nn.Linear, nn.LSTM}
    quantized = torch.quantization.quantize_dynamic(
        model, operators_to_quantize, dtype=dtype
    )
    try:
        logger.info("Applied dynamic PTQ quantization")
        total, trainable = print_model_size(model, model_name="quantized_model")
        logger.info(
            f"Model params after quantize_dynamic (original counts): total={total:,}, trainable={trainable:,}"
        )
    except Exception as e:
        logger.warning("quantize_dynamic: failed to print model size: %s", e)
    return quantized


def prepare_qat(model: nn.Module) -> nn.Module:
    """Prepare a model for Quantization Aware Training (QAT) using default configs.

    This will set `qconfig` on the model and run `prepare_qat` in-place. The caller
    should continue training the model (in train mode) for some epochs, then call
    `convert_qat` to produce a quantized model.
    """
    try:
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.prepare_qat(model, inplace=True)
        logger.info("Model configured for QAT (prepare_qat completed)")
        return model
    except Exception as e:
        raise RuntimeError(
            f"QAT preparation failed - ensure your model supports torch.quantization APIs. Cause: {e}"
        )


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared (and trained) model to a quantized version.

    The model should be in eval mode before conversion.
    """
    try:
        model.eval()
        quantized = torch.quantization.convert(model.eval(), inplace=False)
        logger.info("Converted QAT model to quantized model")
        return quantized
    except Exception as e:
        raise RuntimeError(
            f"QAT convert failed - ensure model was prepared and trained with QAT steps. Cause: {e}"
        )
