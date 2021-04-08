from __future__ import annotations
from typing import Any, Dict, List, Mapping, Tuple, Union
import warnings

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from topocluster.clustering.utils import (
    compute_optimal_assignments,
    l2_centroidal_distance,
)
from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder
from topocluster.utils.numpy_ops import compute_density_map, compute_rips

from .common import Clusterer


class Tomato(Clusterer):

    pers_pairs: Tensor

    def __init__(
        self, k_kde: int = 100, k_rips: int = 15, scale: float = 0.5, threshold: float = 1.0
    ):
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.threshold = threshold

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(self.pers_pairs[:, 0], self.pers_pairs[:, 1], s=15, c="blue")  # type: ignore[arg-type]
        span = np.array([0, 1])
        ax.plot(span, span, c="black", alpha=0.6)  # type: ignore[call-arg]
        ax.fill_between(span, span, 1, interpolate=True, color="grey")
        ax.set_xlabel("Death")
        ax.set_ylabel("Birth")
        ax.set_title("Persistence Diagram")

        return fig

    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        return {}

    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        return None

    def __call__(self, x: Tensor, threshold: float | None = None) -> Tuple[Tensor, Tensor]:
        threshold = self.threshold if threshold is None else threshold
        x_np = x.detach().cpu().numpy()
        x_np = x_np.reshape(x_np.shape[0], -1)
        clusters, pers_pairs = tomato(
            x_np, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, threshold=threshold
        )

        cluster_labels = np.empty(x_np.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k

        self.pers_pairs = torch.as_tensor(pers_pairs, dtype=torch.float32)
        cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.int32)
        centroids = x[list(clusters.keys())]

        soft_labels = l2_centroidal_distance(x=x, centroids=centroids)
        hard_labels = cluster_labels

        return hard_labels, soft_labels


def tomato(
    pc: np.ndarray, k_kde: int, k_rips: int, scale: float, threshold: float
) -> Tuple[Dict[int, np.ndarray[np.int32]], np.ndarray[np.float32]]:
    """Topological mode analysis tool (Chazal et al., 2013).

    Args:
        pc (np.ndarray): The point-cloud to be clustered
        k_kde (int): Number of neighbors to use when computing the density map
        k_rips (int): Number of neighbors to use when computing the Rips graph.
        scale (float): Bandwidth of the kernel used when computing the density map.
        threshold (float): Thresholding parameter (tau in the paper)

    Returns:
        Tuple[Dict[int, np.ndarray[np.int]], np.ndarray[np.float32]]: Clusters and their pers_pairs
    """
    pc = pc.astype(float)
    #  Compute the k-NN KDE
    density_map, _ = compute_density_map(pc, k_kde, scale)
    density_map = density_map.astype(np.float32)
    sorted_idxs = np.argsort(density_map)
    density_map_sorted = density_map[sorted_idxs]
    pc = pc[sorted_idxs]

    _, rips_idxs = compute_rips(pc, k=k_rips)
    entries, pers_pairs = cluster(density_map_sorted, rips_idxs, threshold=threshold)
    if threshold == 1:
        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])
        result = []
        if see.size > 0:
            for i in np.unique(see[:, 0]):
                result.append(
                    [
                        see[np.where(see[:, 0] == i)[0]][0, 0],
                        max(see[np.where(see[:, 0] == i)[0]][:, 1]),
                    ]
                )
            result = np.array(result)
            for key, value in entries.items():
                entries[key] = sorted_idxs[value]
        else:
            warnings.warn("Clustering unsuccessful; consider expanding the VRC neighbourhood.")
        return entries, density_map_sorted[result]
    else:
        for key, value in entries.items():
            entries[key] = sorted_idxs[value]
        return entries, np.array([[0, 0]])


@jit(nopython=True)
def find_entry_idx_by_point(entries: Dict[int, List[int]], point_idx: int) -> np.int64:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return np.int64(index)
    return np.int64(point_idx)


@jit(nopython=True)
def cluster(
    density_map: np.ndarray, rips_idxs: np.ndarray, threshold: float
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    N = density_map.shape[0]
    pers_pairs = np.array([[-1, -1]])
    #  initialize the union-find data-structure with the final index pointing only to itself

    entries = {N - 1: np.array([N - 1])}

    for i in np.arange(N - 2, -1, -1):
        nbr_idxs = rips_idxs[i]
        # compute the upper star Si = {(i, j1), · · · , (i, jk)} of vertex i in R_δ(L);
        us_idxs = nbr_idxs[nbr_idxs > i]
        # check whether vertex i is a local maximum of f within R_δ
        if us_idxs.size == 0:
            entries[i] = np.array([i])  #  create an entry for the local maximum
        else:
            # approximate the gradient of the underlying probability density function by connecting
            # i to its neighbour in the graph with the highest function value
            g_i = np.max(us_idxs)  #  find the maximum index
            # Attach vertex i to the tree t containing g(i)
            e_up = find_entry_idx_by_point(entries, g_i)
            entries[e_up] = np.append(entries[e_up], i)
            entries, pers_pairs_i = merge(
                density_map=density_map,
                entries=entries,
                ref_idx=i,
                e_up=e_up,
                us_idxs=us_idxs,
                threshold=threshold,
            )
            if len(pers_pairs_i) > 1:
                pers_pairs = np.append(pers_pairs, pers_pairs_i, axis=0)

    return entries, pers_pairs


@jit(nopython=True)
def merge(
    density_map: np.ndarray,
    entries: Dict[int, List[int]],
    ref_idx: int,
    e_up: int,
    us_idxs: List[int],
    threshold: float,
) -> Tuple[Dict[int, List[int]], np.ndarray[np.int64]]:
    pers_pairs = np.array([[-1, -1]])

    for idx in us_idxs:
        entry_idx = find_entry_idx_by_point(entries, idx)

        if e_up != entry_idx:
            persistence = density_map[entry_idx] - density_map[ref_idx]

            if persistence < threshold:
                entries[e_up] = np.append(entries[e_up], entries[entry_idx])
                entries.pop(entry_idx)

            pers_pairs = np.append(pers_pairs, np.array([[entry_idx, ref_idx]]), axis=0)

    return entries, pers_pairs
