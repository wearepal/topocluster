from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Dict, Literal, Optional, Tuple, Union

import faiss
import numpy as np
from omegaconf import MISSING
from pykeops.torch import LazyTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from topocluster.clustering.common import Clusterer
from topocluster.clustering.utils import (
    compute_optimal_assignments,
    l2_centroidal_distance,
)
from topocluster.data.data_modules import IGNORE_INDEX

__all__ = ["Kmeans", "run_kmeans_torch", "run_kmeans_faiss"]


class Backends(Enum):
    FAISS = auto()
    TORCH = auto()


class Kmeans(Clusterer):

    labels: Tensor
    centroids: Tensor

    def __init__(
        self,
        n_iter: int,
        k: Optional[int] = None,
        backend: Backends = Backends.FAISS,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iter = n_iter
        self.backend = backend
        self.verbose = verbose

    def build(self, input_dim: int, num_classes: int) -> None:
        self.k = num_classes

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.k is None:
            raise ValueError("Value for k not yet set.")
        if self.backend == "torch":
            hard_labels, centroids = run_kmeans_torch(
                x,
                num_clusters=self.k,
                n_iter=self.n_iter,
                verbose=self.verbose,
            )
        else:
            hard_labels, centroids = run_kmeans_faiss(
                x=x,
                num_clusters=self.k,
                n_iter=self.n_iter,
                verbose=self.verbose,
            )
        soft_labels = l2_centroidal_distance(x=x, centroids=centroids)

        return hard_labels, soft_labels

    def get_loss(
        self, x: Tensor, soft_labels: Tensor, hard_labels: Tensor, y: Tensor, prefix: str = ""
    ) -> Dict[str, Tensor]:
        if prefix:
            prefix += "/"

        labeled = y != IGNORE_INDEX
        _, cluster_map = compute_optimal_assignments(
            labels_pred=hard_labels[labeled].detach().cpu().numpy(),
            labels_true=y[labeled].detach().cpu().numpy(),
            encode=True,
        )
        permute_inds = list(cluster_map.values())
        soft_labels_permuted = soft_labels[labeled][:, permute_inds]
        purity_loss = F.cross_entropy(soft_labels_permuted, y[labeled])
        return {f"{prefix}purity_loss": purity_loss}


def run_kmeans_torch(
    x: torch.Tensor,
    num_clusters: int,
    n_iter: int = 10,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    x = x.flatten(start_dim=1)
    N, D = x.shape  # Number of samples, dimension of the ambient space
    dtype = torch.float32 if x.is_cuda else torch.float64

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:num_clusters, :].clPtone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    cl = None
    print("Finding K means...", flush=True)  # flush to avoid conflict with tqdm
    for _ in tqdm(range(n_iter)):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        # (Npoints, Nclusters) symbolic matrix of squared distances
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    if cl is None:
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

    end = time.time()

    if verbose:
        print(f"K-means with {N:,} points in dimension {D:,}, K = {num_clusters:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                n_iter, end - start, n_iter, (end - start) / n_iter
            )
        )

    return cl, c


def run_kmeans_faiss(
    x: Tensor,
    num_clusters: int,
    n_iter: int,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    x_np = x.detach().cpu().numpy()
    x_np = np.reshape(x_np, (x_np.shape[0], -1))
    n_data, d = x_np.shape

    if x.is_cuda:
        # faiss implementation of k-means
        kmeans = faiss.Clustering(d, num_clusters)
        kmeans.niter = n_iter
        kmeans.max_points_per_centroid = 10000000
        kmeans.verbose = verbose
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        kmeans.train(x_np, index)
        flat_config.device = 0
        _, cluster_indexes = index.search(x_np, 1)
        centroids = faiss.vector_float_to_array(kmeans.centroids)
        centroids = centroids.reshape(num_clusters, d)
    else:
        kmeans = faiss.Kmeans(d=d, k=num_clusters, verbose=verbose, niter=20)
        kmeans.train(x_np)
        _, cluster_indexes = kmeans.index.search(x_np, 1)
        centroids = kmeans.centroids

    cluster_indexes = torch.as_tensor(cluster_indexes, dtype=torch.long, device=x.device).squeeze()
    centroids = torch.as_tensor(centroids, dtype=torch.float32, device=x.device)

    return cluster_indexes, centroids
