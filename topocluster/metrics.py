"""Functions for computing metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from torch import Tensor

from topocluster.metrics import compute_optimal_assignments

__all__ = ["clustering_metrics", "compute_abs_subgroup_id"]


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

    cluster_map = compute_optimal_assignments(labels_true=subgroup_id, labels_pred=preds_np)

    num_hits_all = 0
    for i, (class_id, cluster_id) in enumerate(cluster_map.items()):
        class_mask = subgroup_id == class_id
        num_matches = class_mask.sum()
        if num_matches > 0:
            num_hits = (class_mask & (preds_np == cluster_id)).sum()
            subgroup_acc = num_hits / num_matches
            logging_dict[f"{prefix}Cluster_Acc/{i}"] = subgroup_acc
            num_hits_all += num_hits
    logging_dict[f"{prefix}Cluster_Acc/Total"] = num_hits_all / len(subgroup_id)

    return logging_dict
