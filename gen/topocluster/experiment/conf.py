# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any
from typing import Optional


@dataclass
class ExperimentConf:
    _target_: str = "topocluster.experiment.Experiment"
    datamodule: Any = MISSING  # DataModule
    encoder: Any = MISSING  # Encoder
    clusterer: Any = MISSING  # Clusterer
    trainer: Any = MISSING  # Trainer
    pretrainer: Any = MISSING  # Trainer
    lr: float = 0.001
    weight_decay: float = 9
    log_offline: bool = False
    seed: Optional[int] = 42
    enc_loss_w: float = 1.0
    clust_loss_w: float = 1.0
    exp_group: Optional[str] = None
    train_eval_freq: int = 1
    eval_mode: bool = False
