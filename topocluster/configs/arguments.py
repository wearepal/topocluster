from typing import Any, Dict, List, Literal, Optional

import torch
from ethicml.data import CelebAttrs
from ethicml.data.tabular_data.adult import AdultSplits
from typed_flags import TypedFlags

__all__ = ["BaseArgs", "ClusterArgs"]


class BaseArgs(TypedFlags):
    """General data set settings."""

    dataset: Literal["adult", "cmnist", "celeba"] = "cmnist"

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    biased_train: bool = True  # if True, make the training set biased, dependent on mixing factor
    mixing_factor: float = 0.0  # How much of context should be mixed into training?
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    data_split_seed: int = 888
    root: str = ""
    save_dir: str = "experiments"

    # Dataset manipulation
    missing_s: List[int] = []

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AdultSplits = "Sex"
    drop_discrete: bool = False
    balanced_context: bool = False
    balanced_test: bool = True
    balance_all_quadrants: bool = True

    # Colored MNIST settings
    scale: float = 0.02
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[int, float] = {}
    subsample_train: Dict[int, float] = {}
    input_noise: bool = True  # add uniform noise to the input
    filter_labels: List[int] = []
    colors: List[int] = []

    # CelebA settings
    celeba_sens_attr: CelebAttrs = "Male"
    celeba_target_attr: CelebAttrs = "Smiling"

    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    use_wandb: bool = False

    # Global variables
    _s_dim: int
    _y_dim: int

    #  General training settings
    epochs: int = 250
    gpu: int = 0  # which GPU to use (if available)
    batch_size: int = 256
    test_batch_size: Optional[int] = None
    num_workers: int = 4
    seed: int = 42

    #  Arguments related to topological clustering
    tc_batch_size: Optional[int] = None
    tc_scale: float = 0.5
    tc_k_kde: int = 100
    tc_k_vrc: int = 15
    tc_threshold: float = 1.0
    tc_umap_kwargs: Optional[Dict[str, Any]] = None

    def process_args(self) -> None:
        if not 0 < self.data_pcnt <= 1:
            raise ValueError("data_pcnt has to be between 0 and 1")


class ClusterArgs(BaseArgs):
    """Flags for clustering."""

    # Optimization settings
    early_stopping: int = 30
    eval_on_recon: bool = True

    # Evaluation settings
    eval_epochs: int = 40
    eval_lr: float = 1e-3
    encode_batch_size: int = 1000

    # Training settings
    resume: Optional[str] = None
    evaluate: bool = False
    super_val: bool = False  # Train classifier on encodings as part of validation step.
    super_val_freq: int = 0  # how often to do super val, if 0, do it together with the normal val
    val_freq: int = 5
    log_freq: int = 50
    results_csv: str = ""  # name of CSV file to save results to
    feat_attr: bool = False
    cluster: Literal["s", "y", "both"] = "both"
    with_supervision: bool = True

    # Encoder settings
    encoder: Literal["ae", "vae", "rotnet"] = "ae"
    enc_levels: int = 4
    enc_dim: int = 64
    init_channels: int = 32
    recon_loss: Literal["l1", "l2", "bce", "huber", "ce", "mixed"] = "l2"
    vgg_weight: float = 0
    std_transform: Literal["softplus", "exp"] = "exp"
    kl_weight: float = 1
    elbo_weight: float = 1
    stochastic: bool = False
    enc_path: str = ""
    enc_epochs: int = 100
    enc_lr: float = 1e-3
    enc_wd: float = 0
    enc_wandb: bool = False
    save_encodings: bool = False
    finetune_encoder: bool = False
    finetune_lr: float = 1e-6
    finetune_wd: float = 0
    freeze_layers: int = 0

    # PseudoLabeler
    pseudo_labeler: Literal["ranking", "cosine"] = "ranking"
    sup_ce_weight: float = 1.0
    sup_bce_weight: float = 1.0
    k_num: int = 5
    lower_threshold: float = 0.5
    upper_threshold: float = 0.5

    # Classifier
    cl_hidden_dims: List[int] = [256]
    lr: float = 1e-3
    weight_decay: float = 0
    use_multi_head: bool = False

    # Method
    method: Literal[
        "pl_enc", "pl_output", "pl_enc_no_norm", "kmeans", "topocluster"
    ] = "pl_enc_no_norm"

    _device: torch.device
    _s_dim: int
    _y_dim: int

    #  Labeler
    labeler_lr: float = 1e-3
    labeler_wd: float = 0
    labeler_hidden_dims: List[int] = [100, 100]
    labeler_epochs: int = 100
    labeler_wandb: bool = False

    def process_args(self) -> None:
        super().process_args()
        if self.super_val_freq < 0:
            raise ValueError("frequency cannot be negative")
