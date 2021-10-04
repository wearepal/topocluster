from __future__ import annotations
from functools import partial
from typing import NamedTuple, Union, cast, overload

from pykeops.torch import LazyTensor  # type: ignore
import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import Literal

__all__ = ["knn", "pnorm", "rbf", "cosine_similarity", "Kernel"]


TensorType = Union[LazyTensor, Tensor]


def _l2_normalize(tensor: TensorType, dim: int):
    norm = (tensor ** 2).sum(dim) ** 2  # type: ignore
    return tensor / norm


def cosine_similarity(
    tensor_a: TensorType,
    tensor_b: TensorType,
    *,
    dim: int = -1,
    normalize: bool = True,
) -> Tensor:
    if normalize:
        tensor_a = _l2_normalize(tensor_a, dim=-1)
        tensor_b = _l2_normalize(tensor_b, dim=-1)

    return cast(Tensor, (tensor_a * tensor_b).sum(dim))


NormType = Union[int, Literal["inf", "sup"]]


def pnorm(
    tensor_a: TensorType,
    tensor_b: TensorType,
    *,
    dim: int = -1,
    p: NormType = 2,
    root: bool = True,
) -> Tensor:
    dists = (tensor_a - tensor_b).abs()
    if isinstance(p, int):
        if p < 1:
            raise ValueError("If 'p' is an integer, it must be positive.")
        norm = (dists ** p).sum(dim)
        if root:
            norm = norm ** (1 / p)  # type: ignore
        return norm  # type: ignore
    elif p == "inf":
        res = dists.min(dim)
    else:
        res = dists.max(dim)
    # torch.max returns a named tuple of (values, indices)
    if isinstance(res, tuple):
        return res.values  # type: ignore
    return res  # type: ignore


def rbf(x: Tensor, y: Tensor, *, scale: float, dim: int = 1) -> Tensor:
    return torch.exp(pnorm(x, y, p=2, root=False, dim=dim) / scale)


class KnnOutput(NamedTuple):
    indices: Tensor
    distances: Tensor


Kernel = Literal["pnorm", "cosine"]


@overload
def knn(
    pc: Tensor,
    k: int,
    kernel: Kernel = ...,
    backend: Literal["keops", "torch"] = "torch",
    p: NormType = 2,
    normalize: bool = True,
    return_distances: Literal[False] = ...,
) -> Tensor:
    ...


@overload
def knn(
    pc: Tensor,
    k: int,
    kernel: Kernel = ...,
    backend: Literal["keops", "torch"] = "torch",
    p: NormType = 2,
    normalize: bool = True,
    return_distances: Literal[True] = ...,
) -> KnnOutput:
    ...


def knn(
    pc: Tensor,
    k: int,
    kernel: Kernel = "pnorm",
    backend: Literal["keops", "torch"] = "torch",
    p: NormType = 2,
    normalize: bool = True,
    return_distances: bool = False,
) -> Tensor | KnnOutput:

    if kernel == "cosine":
        kernel_fn = partial(cosine_similarity, normalize=False)
        if normalize:
            pc = F.normalize(pc, dim=1, p=2)
    else:
        kwargs = {"p": p}
        if normalize:
            kwargs["root"] = True
        kernel_fn = partial(pnorm, **kwargs)

    G_i = pc[:, None]  # (M**2, 1, 2)
    X_j = pc[None]  # (1, N, 2)
    distances = None

    if backend == "keops":
        G_i_lt = LazyTensor(G_i)  # (M**2, 1, 2)
        X_j_lt = LazyTensor(X_j)  # (1, N, 2)
        # symbolic matrix of squared distances
        D_ij = kernel_fn(G_i_lt, X_j_lt)  # (M**2, N)
        # Grid <-> Samples, (M**2, K) integer tensor
        indices = D_ij.argKmin(k, dim=1)  # type: ignore
        if return_distances:
            # Workaround for pykeops currently not supporting differentiation through Kmin
            distances = kernel_fn(pc[:, None], pc[indices, :])
    else:
        # brute-force approach using torch.topk
        D_ij = kernel_fn(G_i, X_j)
        values_indices = D_ij.topk(k, dim=1, largest=False)
        indices = values_indices.indices
        if return_distances:
            distances = values_indices.values

    if distances is None:
        return indices
    return KnnOutput(indices=indices, distances=distances)
