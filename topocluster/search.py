from __future__ import annotations
from abc import abstractmethod
import math
from typing import Generic, NamedTuple, TypeVar, Union, overload

import attr
import faiss  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal

__all__ = [
    "Knn",
    "KnnExact",
    "KnnIVF",
    "KnnIVFPQ",
    "pnorm",
]


def pnorm(
    tensor_a: Tensor,
    tensor_b: Tensor,
    *,
    p: float = 2,
    root: bool = True,
    dim: int = -1,
) -> Tensor:
    dists = (tensor_a - tensor_b).abs()
    if math.isinf(p):
        if p > 0:
            norm = dists.max(dim).values
        else:
            norm = dists.min(dim).values
    else:
        norm = (dists ** p).sum(dim)
        if root:
            norm = norm ** (1 / p)  # type: ignore
    return norm


T = TypeVar("T", Tensor, npt.NDArray[np.floating])


class KnnOutput(NamedTuple):
    indices: Tensor | npt.NDArray[np.uint]
    distances: Tensor | npt.NDArray[np.floating]


@attr.define(kw_only=True)
class Knn(nn.Module):
    k: int
    p: float = 2
    root: bool = False
    normalize: bool = False
    """
    Whether to Lp-normalize the vectors for pairwise-distance computation.
    .. note:: 
        When vectors u and v are normalized to unit length, the Euclidean distance betwen them
        is equal to :math:`\\|u - v\\|^2 = (1-\\cos(u, v))`, that is the Euclidean distance over the end-points
        of u and v is a proper metric which gives the same ordering as the Cosine distance for
        any comparison of vectors, and furthermore avoids the potentially expensive trigonometric
        operations required to yield a proper metric.
    """

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _build_index(self, d: int) -> faiss.IndexFlat:
        ...

    def _index_to_gpu(self, index: faiss.IndexFlat) -> faiss.GpuIndexFlat:  # type: ignore
        # use a single GPU
        res = faiss.StandardGpuResources()  # type: ignore
        # make it a flat GPU index
        return faiss.index_cpu_to_gpu(res, x.device.index, index)  # type: ignore

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = ...,
        return_distances: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = ...,
        return_distances: Literal[True] = ...,
    ) -> KnnOutput:
        ...

    def forward(
        self,
        x: Tensor,
        *,
        y: Tensor | None = None,
        return_distances: bool = False,
    ) -> Tensor | KnnOutput:

        x_np = x.detach().cpu().numpy()
        if self.normalize:
            x = F.normalize(x, dim=1, p=self.p)

        if y is None:
            y = x
            y_np = x_np
        else:
            if self.normalize:
                y = F.normalize(y, dim=1, p=self.p)
            y_np = x.detach().cpu().numpy()

        index = self._build_index(d=x.size(1))
        if x.is_cuda or y.is_cuda:
            index = self._index_to_gpu(index=index)

        if not index.is_trained:
            index.train(x=x_np)  # type: ignore
        # add vectors to the index
        index.add(x=x_np)  # type: ignore
        # search for the nearest k neighbors for each data-point
        distances_np, indices_np = index.search(x=y_np, k=self.k)  # type: ignore
        # Convert back from numpy to torch
        indices = torch.as_tensor(indices_np, device=x.device)

        if return_distances:
            if x.requires_grad or y.requires_grad:
                distances = pnorm(x[:, None], y[indices, :], dim=-1, p=self.p, root=False)
            else:
                distances = torch.as_tensor(distances_np, device=x.device)

            # Take the root of the distances to 'complete' the norm
            if self.root and (not math.isinf(self.p)):
                distances = distances ** (1 / self.p)

            return KnnOutput(indices=indices, distances=distances)
        return indices


@attr.define(kw_only=True)
class KnnExact(Knn):
    def _build_index(self, d: int) -> faiss.IndexFlat:
        index = faiss.IndexFlat(d, faiss.METRIC_Lp)
        index.metric_arg = self.p
        return index


@attr.define(kw_only=True)
class KnnIVF(KnnExact):
    nlist: int = 100
    """Number of Voronoi cells to form with k-means clustering."""
    nprobe: int = 1
    """Number of neighboring Voronoi cells to probe."""

    def _build_index(self, d: int) -> faiss.IndexIVFFlat:
        quantizer = super()._build_index(d=d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        index.nprobe = self.nprobe
        return index


@attr.define(kw_only=True)
class KnnIVFPQ(KnnExact):
    nlist: int = 100
    """Number of Voronoi cells to form with k-means clustering."""
    nprobe: int = 1
    """Number of neighboring Voronoi cells to probe."""
    bits: int = 8
    num_centroids = 8

    def _build_index(self, d: int) -> faiss.IndexIVFPQ:
        quantizer = super()._build_index(d=d)
        m = d // self.num_centroids
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, m, self.bits)
        index.nprobe = self.nprobe
        return index
