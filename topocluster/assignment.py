from __future__ import annotations
from torch import Tensor

from lapjv import lapjv
import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "optimal_assignment",
    "compute_cost_matrix",
    "encode_arr_with_dict",
]


def optimal_assignment(
    labels_pred: np.ndarray | Tensor,
    *,
    labels_true: np.ndarray | Tensor,
    num_classes: int | None = None,
    encode: bool = True,
) -> dict[int, int]:
    """Find an assignment of cluster to class such that the overall accuracy is maximized."""
    # row_ind maps from class ID to cluster ID: cluster_id = row_ind[class_id]
    # col_ind maps from cluster ID to class ID: class_id = row_ind[cluster_id]
    cost_matrix, decodings_pred, decodings_true = compute_cost_matrix(
        labels_pred=labels_pred, labels_true=labels_true, num_classes=num_classes, encode=encode
    )

    if cost_matrix.shape[0] == cost_matrix.shape[1]:
        _, col_ind, _ = lapjv(-cost_matrix)
    else:
        _, col_ind = linear_sum_assignment(-cost_matrix)
    label_map = {}
    for label_true, label_pred in enumerate(col_ind):
        if decodings_true is not None:
            label_true = decodings_true[label_true]
        if decodings_pred is not None:
            label_pred = decodings_pred[label_pred]
        label_map[label_pred] = label_true

    return label_map


def _get_index_mapping(arr: np.ndarray | Tensor) -> tuple[dict[int, int], dict[int, int]]:
    encodings, decodings = {}, {}
    for i, val in enumerate(np.unique(arr)):
        encodings[val] = i
        decodings[i] = val
    return encodings, decodings


def encode_arr_with_dict(arr: np.ndarray | Tensor, *, encoding_dict: dict[int, int]) -> np.ndarray:
    return np.vectorize(encoding_dict.__getitem__)(arr)


def compute_cost_matrix(
    labels_pred: np.ndarray | Tensor,
    *,
    labels_true: np.ndarray | Tensor,
    num_classes: int | None = None,
    encode: bool = True,
) -> tuple[np.ndarray, dict[int, int] | None, dict[int, int] | None]:
    decodings_true, decodings_pred = None, None
    if encode and num_classes is None:
        encodings_pred, decodings_pred = _get_index_mapping(labels_pred)
        encodings_true, decodings_true = _get_index_mapping(labels_true)
        labels_pred = encode_arr_with_dict(labels_pred, encoding_dict=encodings_pred)
        labels_true = encode_arr_with_dict(labels_true, encoding_dict=encodings_true)
        cost_matrix = np.zeros((len(encodings_true), len(encodings_pred)))
    elif num_classes is None:
        cost_matrix = np.zeros((len(np.unique(labels_true)), len(np.unique(labels_pred))))
    else:
        cost_matrix = np.zeros((num_classes, num_classes))

    indices, counts = np.unique(np.stack([labels_true, labels_pred]), axis=1, return_counts=True)
    cost_matrix[tuple(indices)] += counts

    return cost_matrix, decodings_pred, decodings_true
