# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class AutoEncoderConf:
    _target_: str = "topocluster.models.autoencoder.AutoEncoder"
    latent_dim: int = MISSING
    lr: float = 0.001


@dataclass
class GatedConvAutoEncoderConf:
    _target_: str = "topocluster.models.autoencoder.GatedConvAutoEncoder"
    init_hidden_dims: int = MISSING
    levels: int = MISSING
    latent_dim: int = MISSING
    lr: float = 0.001
