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


def quantize_dynamic_ptq(model, dtype=torch.qint8, operators_to_quantize=None):
    if operators_to_quantize is None:
        operators_to_quantize = {nn.Linear, nn.LSTM}

    # Build a module-level qconfig_spec: enable qconfig for Linear/LSTM,
    # but DISABLE for Linear layers inside MultiheadAttention.
    from torch.quantization import default_dynamic_qconfig

    qconfig_spec = {}
    for name, module in model.named_modules():
        if isinstance(module, tuple(operators_to_quantize)):
            # Skip modules that are part of MultiheadAttention
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = dict(model.named_modules()).get(parent_name, None)
            if isinstance(parent, nn.MultiheadAttention):
                continue
            qconfig_spec[name] = default_dynamic_qconfig

    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec=qconfig_spec, dtype=dtype
    )

    for m in quantized.modules():
        if isinstance(m, nn.TransformerEncoderLayer):
            m.activation_relu_or_gelu = 0

    logger.info("Applied dynamic PTQ (skipping MultiheadAttention internals)")
    return quantized


def prepare_qat(model: nn.Module) -> nn.Module:
    try:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        # Disable quantization for modules incompatible with QAT
        for m in model.modules():
            # bỏ qua MultiheadAttention
            if isinstance(m, nn.MultiheadAttention):
                m.qconfig = None
            # bỏ qua embedding
            if isinstance(m, nn.Embedding):
                m.qconfig = None
            # Bỏ qua layernorm
            if isinstance(m, nn.LayerNorm):
                m.qconfig = None

        # Disable quantization for CRF layers if present
        if hasattr(model, "crf"):
            for m in model.crf.modules():
                m.qconfig = None

        torch.quantization.prepare_qat(model, inplace=True)
        logger.info("Model configured for QAT (prepare_qat completed)")
        return model
    except Exception as e:
        raise RuntimeError(f"QAT preparation failed: {e}")


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared (and trained) model to a quantized version.

    The model should be in eval mode before conversion.
    """
    try:
        model.eval()
        quantized = torch.quantization.convert(model.eval(), inplace=False)

        for m in quantized.modules():
            if isinstance(m, nn.TransformerEncoderLayer):
                m.activation_relu_or_gelu = 0

        logger.info("Converted QAT model to quantized model")
        return quantized
    except Exception as e:
        raise RuntimeError(
            f"QAT convert failed - ensure model was prepared and trained with QAT steps. Cause: {e}"
        )
