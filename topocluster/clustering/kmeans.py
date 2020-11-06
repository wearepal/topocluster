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
        cuda: bool = False,
        backend: Backends = Backends.FAISS,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iter = n_iter
        self.cuda = cuda
        self.backend = backend
        self.verbose = verbose

    def build(self, input_dim: int, num_classes: int) -> None:
        self.k = num_classes

    def get_loss(self, x: Tensor, y: Tensor, prefix: str = "") -> Dict[str, Tensor]:
        if prefix:
            prefix += "/"

        labeled = y != -1
        _, cluster_map = compute_optimal_assignments(
            labels_pred=self.hard_labels[labeled].cpu().detach().numpy(),
            labels_true=y[labeled].cpu().detach().numpy(),
            encode=True,
        )
        permute_inds = list(cluster_map.values())
        soft_labels_permuted = self.soft_labels[labeled][:, permute_inds]
        purity_loss = F.cross_entropy(soft_labels_permuted, y[labeled])
        return {f"{prefix}purity_loss": purity_loss}

    def fit(self, x: Tensor) -> Kmeans:
        if self.k is None:
            raise ValueError("Value for k not yet set.")
        if self.backend == "torch":
            self.hard_labels, centroids = run_kmeans_torch(
                x,
                k=self.k,
                device=torch.device("cuda") if self.cuda else torch.device("cpu"),
                n_iter=self.n_iter,
                verbose=self.verbose,
            )
        else:
            self.hard_labels, centroids = run_kmeans_faiss(
                x=x, nmb_clusters=self.k, n_iter=self.n_iter, cuda=self.cuda, verbose=self.verbose
            )

        self.soft_labels = l2_centroidal_distance(x=x, centroids=centroids)

        return self


def run_kmeans_torch(
    x: torch.Tensor,
    k: int,
    device: torch.device,
    n_iter: int = 10,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    x = x.flatten(start_dim=1)
    N, D = x.shape  # Number of samples, dimension of the ambient space
    dtype = torch.float64 if device.type == "cpu" else torch.float32

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:k, :].clPtone()  # Simplistic random initialization
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
        print(f"K-means with {N:,} points in dimension {D:,}, K = {k:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                n_iter, end - start, n_iter, (end - start) / n_iter
            )
        )

    return cl, c


def run_kmeans_faiss(
    x: Union[np.ndarray, Tensor], nmb_clusters: int, n_iter: int, cuda: bool, verbose: bool = False
) -> Tuple[Tensor, Tensor]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.reshape(x, (x.shape[0], -1))
    n_data, d = x.shape

    if cuda:
        # faiss implementation of k-means
        kmeans = faiss.Clustering(d, nmb_clusters)
        kmeans.niter = n_iter
        kmeans.max_points_per_centroid = 10000000
        kmeans.verbose = verbose
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        kmeans.train(x, index)
        flat_config.device = 0
        D, I = index.search(x, 1)
    else:
        kmeans = faiss.Kmeans(d=d, k=nmb_clusters, verbose=verbose, niter=20)
        kmeans.train(x)
        D, I = kmeans.index.search(x, 1)

    I = torch.as_tensor(I, dtype=torch.long).squeeze()
    centroids = torch.as_tensor(kmeans.centroids)

    return I, centroids
