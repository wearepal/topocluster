from __future__ import annotations
from functools import partial
import math
from typing import NamedTuple, cast, overload

import attr
import faiss  # type: ignore
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal

__all__ = [
    "KnnExact",
    "KnnIVF",
    "KnnIVFPQ",
    "pnorm",
    "cosine_similarity",
    "Kernel",
]


def cosine_similarity(
    tensor_a: Tensor,
    tensor_b: Tensor,
    *,
    dim: int = -1,
    normalize: bool = True,
) -> Tensor:
    if normalize:
        tensor_a = F.normalize(tensor_a, dim=dim, p=2)
        tensor_b = F.normalize(tensor_b, dim=dim, p=2)

    return cast(Tensor, (tensor_a * tensor_b).sum(dim))


def pnorm(
    tensor_a: Tensor,
    tensor_b: Tensor,
    *,
    dim: int = -1,
    p: float,
    root: bool = True,
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


class KnnOutput(NamedTuple):
    indices: Tensor
    distances: Tensor


Kernel = Literal["pnorm", "cosine"]


@attr.define(kw_only=True)
class KnnExact(nn.Module):
    # import faiss

    k: int
    kernel: Kernel = "pnorm"
    p: float = 2
    root: bool = False

    def __attrs_pre_init__(self):
        super().__init__()

    def _build_index(self, d: int) -> faiss.IndexFlat:
        if self.kernel == "cosine":
            index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlat(d, faiss.METRIC_Lp)
            index.metric_arg = self.p
        return index

    def _index_to_gpu(self, index: faiss.IndexFlat) -> faiss.GpuIndexFlat:  # type: ignore
        # use a single GPU
        res = faiss.StandardGpuResources()  # type: ignore
        # make it a flat GPU index
        return faiss.index_cpu_to_gpu(res, x.device.index, index)  # type: ignore

    @overload
    def forward(
        self,
        x: Tensor,
        return_distances: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        return_distances: Literal[True] = ...,
    ) -> KnnOutput:
        ...

    def forward(
        self,
        x: Tensor,
        return_distances: bool = False,
    ) -> Tensor | KnnOutput:
        d = x.size(1)
        index = self._build_index(d=d)

        if x.is_cuda:
            index = self._index_to_gpu(index=index)

        if self.kernel == "cosine":
            x = F.normalize(x, dim=1, p=2)

        x_np = x.detach().cpu().numpy()
        if not index.is_trained:
            index.train(x=x_np)  # type: ignore
        # add vectors to the index
        index.add(x=x_np)  # type: ignore
        # search for the nearest k neighbors for each data-point
        distances_np, indices_np = index.search(x=x_np, k=self.k)  # type: ignore
        # Convert back from numpy to torch
        indices = torch.as_tensor(indices_np, device=x.device)

        if return_distances:
            if x.requires_grad:
                if self.kernel == "cosine":
                    kernel_fn = partial(cosine_similarity, normalize=False)
                else:
                    kernel_fn = partial(pnorm, p=self.p, root=False)
                distances = kernel_fn(x[:, None], x[indices, :], dim=-1)
            else:
                distances = torch.as_tensor(distances_np, device=x.device)

            # Take the root of the distances to 'complete' the norm
            if (self.kernel == "pnorm") and self.root and (not math.isinf(self.p)):
                distances = distances ** (1 / self.p)

            return KnnOutput(indices=indices, distances=distances)
        return indices


@attr.define(kw_only=True)
class KnnIVF(KnnExact):
    ivf: bool = False
    nlist: int = 100
    nprobe: int = 1

    def _build_index(self, d: int) -> faiss.IndexIVFFlat:
        quantizer = super()._build_index(d=d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        index.nprobe = self.nprobe
        return index


@attr.define(kw_only=True)
class KnnIVFPQ(KnnExact):
    nlist: int = 100
    bits: int = 8
    num_centroids = 8
    nprobe: int = 1

    def _build_index(self, d: int) -> faiss.IndexIVFPQ:
        quantizer = super()._build_index(d=d)
        m = d // self.num_centroids
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, m, self.bits)
        index.nprobe = self.nprobe
        return index
