from __future__ import annotations
import time
from typing import Optional

from pykeops.torch import LazyTensor
import torch
from torch import Tensor
import torch.nn.functional as F

from topocluster.clustering.common import Clusterer
from topocluster.clustering.utils import (
    compute_optimal_assignments,
    l2_centroidal_distance,
)
from topocluster.data.utils import IGNORE_INDEX

__all__ = ["Kmeans", "run_kmeans"]


class Kmeans(Clusterer):
    def __init__(
        self,
        n_iter: int,
        k: Optional[int] = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iter = n_iter
        self.verbose = verbose

    def build(self, input_dim: int, num_classes: int) -> None:
        self.k = num_classes

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.k is None:
            raise ValueError("Value for 'k' not yet set.")
        hard_labels, centroids = run_kmeans(
            x,
            num_clusters=self.k,
            n_iter=self.n_iter,
            verbose=self.verbose,
        )
        hard_labels = hard_labels.detach()
        centroids = centroids.detach()
        soft_labels = l2_centroidal_distance(x=x, centroids=centroids)

        return hard_labels, soft_labels

    def get_loss(
        self, x: Tensor, soft_labels: Tensor, hard_labels: Tensor, y: Tensor, prefix: str = ""
    ) -> dict[str, Tensor]:
        if prefix:
            prefix += "/"

        labeled = y != IGNORE_INDEX
        y_l = y[labeled]
        soft_labels_l = soft_labels[labeled]
        hard_labels_l = hard_labels[labeled]

        _, cluster_map = compute_optimal_assignments(
            labels_pred=hard_labels_l.detach().cpu().numpy(),
            labels_true=y_l.detach().cpu().numpy(),
            num_classes=self.k,
            encode=False,
        )

        mask = torch.zeros_like(y_l, dtype=torch.bool)
        mapped_inds = torch.empty_like(y_l, dtype=torch.long)
        for class_ind, cluster_ind in cluster_map.items():
            mask_k = (y_l == class_ind) & (hard_labels_l == cluster_ind)
            mapped_inds[mask_k] = cluster_ind
            mask |= mask_k

        softmax_xent = -torch.mean(
            F.log_softmax(soft_labels_l[mask], dim=-1).gather(1, mapped_inds[mask].view(-1, 1))
        )
        return {f"{prefix}purity_loss": softmax_xent}


def run_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    n_iter: int = 10,
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    x = x.flatten(start_dim=1)
    N, D = x.shape  # Number of samples, dimension of the ambient space
    dtype = torch.float32 if x.is_cuda else torch.float64

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    centroids = x[:num_clusters, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    cluster_indexes = None

    for _ in range(n_iter):
        c_j = LazyTensor(centroids[None, :, :])  # (1, Nclusters, D)
        # (Npoints, Nclusters) symbolic matrix of squared distances
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cluster_indexes = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cluster_indexes).type(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            centroids[:, d] = torch.bincount(cluster_indexes, weights=x[:, d]) / Ncl

    if cluster_indexes is None:
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cluster_indexes = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

    end = time.time()

    if verbose:
        print(f"K-means with {N:,} points in dimension {D:,}, K = {num_clusters:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                n_iter, end - start, n_iter, (end - start) / n_iter
            )
        )

    return cluster_indexes, centroids
