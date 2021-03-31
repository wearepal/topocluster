# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from omegaconf import MISSING
from topocluster.clustering.kmeans import Backends
from typing import Any
from typing import Optional


@dataclass
class TomatoConf:
    _target_: str = "topocluster.clustering.Tomato"
    k_kde: int = 100
    k_rips: int = 15
    scale: float = 0.5
    threshold: float = 1.0


@dataclass
class GMMConf:
    _target_: str = "topocluster.clustering.GMM"
    n_components: Any = 1
    covariance_type: Any = "full"
    tol: Any = 0.001
    reg_covar: Any = 1e-06
    max_iter: Any = 100
    n_init: Any = 1
    init_params: Any = "kmeans"
    weights_init: Any = None
    means_init: Any = None
    precisions_init: Any = None
    random_state: Any = None
    warm_start: Any = False
    verbose: Any = 0
    verbose_interval: Any = 10


@dataclass
class TopoGradConf:
    _target_: str = "topocluster.clustering.TopoGrad"
    k_kde: int = MISSING
    k_rips: int = MISSING
    scale: float = MISSING
    threshold: float = MISSING


@dataclass
class KmeansConf:
    _target_: str = "topocluster.clustering.Kmeans"
    n_iter: int = MISSING
    k: Optional[int] = None
    backend: Backends = Backends.FAISS
    verbose: bool = False
