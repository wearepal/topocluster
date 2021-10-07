"""Functions for computing metrics."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from torch import Tensor

from topocluster.assignment import optimal_assignment

__all__ = [
    "clustering_accuracy",
    "clustering_metrics",
    "compute_abs_subgroup_id",
]


def clustering_accuracy(
    labels_true: npt.NDArray[np.int_] | Tensor, labels_pred: npt.NDArray[np.int_] | Tensor
) -> float:
    label_map = optimal_assignment(labels_true=labels_true, labels_pred=labels_pred, encode=True)
    num_hits = 0
    for label_pred, label_true in label_map.items():
        num_hits += ((labels_true == label_true) & (labels_pred == label_pred)).sum()
    return float(num_hits / len(labels_true) * 100)


def compute_abs_subgroup_id(
    superclass_inf: Tensor | np.ndarray, *, subgroup_inf: Tensor | np.ndarray, num_subgroups: int
) -> Tensor | np.ndarray:
    return superclass_inf * num_subgroups + subgroup_inf


def clustering_metrics(
    preds: Tensor,
    *,
    subgroup_inf: Tensor,
    superclass_inf: Tensor,
    num_subgroups: int,
    prefix: str = "",
) -> dict[str, float]:
    # Convert from torch to numpy
    preds_np = preds.detach().cpu().numpy()
    superclass_inf_np = superclass_inf.cpu().numpy()
    subgroup_inf_np = subgroup_inf.cpu().numpy()

    subgroup_id = compute_abs_subgroup_id(
        superclass_inf=superclass_inf_np, subgroup_inf=subgroup_inf_np, num_subgroups=num_subgroups
    )

    if prefix:
        prefix += "/"
    logging_dict = {
        f"{prefix}ARI": adjusted_rand_score(labels_true=subgroup_id, labels_pred=preds_np),
        f"{prefix}AMI": adjusted_mutual_info_score(labels_true=subgroup_id, labels_pred=preds_np),  # type: ignore
        f"{prefix}NMI": normalized_mutual_info_score(labels_true=subgroup_id, labels_pred=preds_np),  # type: ignore
    }

    cluster_map = optimal_assignment(labels_true=subgroup_id, labels_pred=preds_np)

    num_hits_all = 0
    for i, (cluster_id, class_id) in enumerate(cluster_map.items()):
        class_mask = subgroup_id == class_id
        num_matches = class_mask.sum()
        if num_matches > 0:
            num_hits = (class_mask & (preds_np == cluster_id)).sum()
            subgroup_acc = num_hits / num_matches
            logging_dict[f"{prefix}Cluster_Acc/{i}"] = subgroup_acc
            num_hits_all += num_hits
    logging_dict[f"{prefix}Cluster_Acc/Total"] = num_hits_all / len(subgroup_id)

    return logging_dict
