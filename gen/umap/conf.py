# Generated by configen, do not edit.
# See https://github.com/facebookresearch/hydra/tree/master/tools/configen
# fmt: off
# isort:skip_file
# flake8: noqa

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UMAPConf:
    _target_: str = "umap.UMAP"
    n_neighbors: Any = 15
    n_components: Any = 2
    metric: Any = "euclidean"
    metric_kwds: Any = None
    output_metric: Any = "euclidean"
    output_metric_kwds: Any = None
    n_epochs: Any = None
    learning_rate: Any = 1.0
    init: Any = "spectral"
    min_dist: Any = 0.1
    spread: Any = 1.0
    low_memory: Any = True
    n_jobs: Any = -1
    set_op_mix_ratio: Any = 1.0
    local_connectivity: Any = 1.0
    repulsion_strength: Any = 1.0
    negative_sample_rate: Any = 5
    transform_queue_size: Any = 4.0
    a: Any = None
    b: Any = None
    random_state: Any = None
    angular_rp_forest: Any = False
    target_n_neighbors: Any = -1
    target_metric: Any = "categorical"
    target_metric_kwds: Any = None
    target_weight: Any = 0.5
    transform_seed: Any = 42
    transform_mode: Any = "embedding"
    force_approximation_algorithm: Any = False
    verbose: Any = False
    unique: Any = False
    densmap: Any = False
    dens_lambda: Any = 2.0
    dens_frac: Any = 0.3
    dens_var_shift: Any = 0.1
    output_dens: Any = False
    disconnection_distance: Any = None
