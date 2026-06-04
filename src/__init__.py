"""Vietnamese Named Entity Recognition pipeline source package.

Provides configurations, dataset loaders, models, trainers, utils, and quantization tools.
"""

from .config import (
    TRAIN_FILE,
    DEV_FILE,
    TEST_FILE,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    get_model_dirs,
    TransformerConfig,
    BERTConfig,
    LSTMConfig,
    LoRAConfig,
    KDConfig,
    QuantConfig,
)

from .dataset import (
    read_conll,
    VietnameseNERDataset,
    get_dataloader,
)

from .model import (
    CRFLayer,
    LSTMModel,
    BiLSTMModel,
    TransformerModel,
    PhoBERTModel,
    PhoBERTLoRA,
    get_model,
    NERLoss,
)

from .trainer import (
    BaseTrainer,
    DistillationTrainer,
    QuantizationTrainer,
)

from .quantize_utils import (
    quantize_dynamic_ptq,
    prepare_qat,
    convert_qat,
)

from .utils import (
    set_seed,
    setup_logger,
    compute_metrics,
    compute_entity_level_metrics,
    plot_loss_curve,
    plot_acc_curve,
    plot_f1_curve,
    plot_confusion_matrix,
    plot_entity_distribution,
    save_metrics_csv,
    save_predictions,
    get_error_analysis,
    print_model_size,
)

from .data_augmentor import NERDataAugmentor

from .engine import (
    run_evaluation,
    run_train,
    run_evaluate,
    run_distill,
    run_infer,
    run_quantize,
)
