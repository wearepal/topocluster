from __future__ import annotations
from functools import partial
import math
from typing import NamedTuple, cast, overload

import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import Literal

__all__ = ["knn", "pnorm", "cosine_similarity", "Kernel"]


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


@overload
def knn(
    x: Tensor,
    k: int,
    kernel: Kernel = ...,
    return_distances: Literal[False] = ...,
    grad_enabled: bool = ...,
    p: float = 2,
    root: bool = False,
) -> Tensor:
    ...


@overload
def knn(
    x: Tensor,
    k: int,
    kernel: Kernel = ...,
    return_distances: Literal[True] = ...,
    grad_enabled: bool = ...,
    p: float = 2,
    root: bool = False,
) -> KnnOutput:
    ...


def knn(
    x: Tensor,
    k: int,
    kernel: Kernel = "pnorm",
    return_distances: bool = False,
    grad_enabled: bool = True,
    p: float = 2,
    root: bool = False,
) -> Tensor | KnnOutput:
    import faiss

    d = x.size(1)
    if kernel == "cosine":
        x = F.normalize(x, dim=1, p=2)
        index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexFlat(d, faiss.METRIC_Lp)
        index.metric_arg = p
    if x.is_cuda:
        # use a single GPU
        res = faiss.StandardGpuResources()  # type: ignore
        # make it a flat GPU index
        index = faiss.index_cpu_to_gpu(res, x.device.index, index)  # type: ignore

    x_np = x.detach().cpu().numpy()
    # add vectors to the index
    index.add(x=x_np) # type: ignore
    # search for the nearest k neighbors for each data-point
    distances_np, indices_np = index.search(x=x_np, k=k)  # type: ignore
    # Convert back from numpy to torch
    indices = torch.as_tensor(indices_np, device=x.device)

    if return_distances:
        if grad_enabled:
            if kernel == "cosine":
                kernel_fn = partial(cosine_similarity, normalize=False)
            else:
                kernel_fn = partial(pnorm, p=p, root=False)
            distances = kernel_fn(x[:, None], x[indices, :], dim=-1)
        else:
            distances = torch.as_tensor(distances_np, device=x.device)

        # Take the root of the distances to 'complete' the norm
        if (kernel == "pnorm") and root and (not math.isinf(p)):
            distances = distances ** (1 / p)

        return KnnOutput(indices=indices, distances=distances)
    return indices
