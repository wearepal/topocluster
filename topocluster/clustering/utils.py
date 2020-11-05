from __future__ import annotations
from typing import Dict, Tuple

from lapjv import lapjv
import numpy as np
import torch
from torch import Tensor

__all__ = ["l2_centroidal_distance", "find_optimal_assignments"]


def l2_centroidal_distance(x: Tensor, centroids: Tensor):
    return torch.square(centroids - x[:, None]).sum(-1)


def find_optimal_assignments(counts: np.ndarray, num_total: int) -> Tuple[float, Dict[int, int]]:
    """Find an assignment of cluster to class such that the overall accuracy is maximized."""
    # row_ind maps from class ID to cluster ID: cluster_id = row_ind[class_id]
    # col_ind maps from cluster ID to class ID: class_id = row_ind[cluster_id]
    row_ind, _, result = lapjv(-counts)
    best_acc = -result[0] / num_total
    assignments = {class_id: cluster_id for class_id, cluster_id in enumerate(row_ind)}
    return best_acc, assignments


def count_occurances(
    preds: np.ndarray[np.int], true: np.ndarray[np.int], num_classes: int
) -> np.ndarray[np.int]:
    counts = np.zeros((num_classes,) * 2, dtype=np.int64)
    indices, batch_counts = np.unique(np.stack([true, preds]), axis=1, return_counts=True)
    counts[tuple(indices)] += batch_counts
    return counts
