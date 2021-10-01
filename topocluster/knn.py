from __future__ import annotations
from typing import Any, Callable, NamedTuple, Union, overload

from pykeops.torch import LazyTensor
import torch
from torch import Tensor
from typing_extensions import Literal, Protocol

__all__ = [
    "knn",
    "pnorm",
    "rbf",
]


TensorType = Union[LazyTensor, Tensor]


def pnorm(
    tensor_a: TensorType,
    tensor_b: TensorType,
    *,
    dim: int = -1,
    p: int | Literal["inf", "sup"] = 2,
    root: bool = True,
) -> Tensor:
    dists = tensor_a - tensor_b
    if isinstance(p, int):
        if p < 1:
            raise ValueError("If 'p' is an integer, it must be positive.")
        if p == 1:
            return dists.abs().sum(dim)
        norm = (dists ** p).sum(dim)
        if root:
            norm = norm * (1 / p)
        return norm
    elif p == "inf":
        res = dists.min(dim)
    else:
        res = dists.max(dim)
    # torch.max returns a named tuple of (values, indices)
    if isinstance(res, tuple):
        return res.values
    return res


def rbf(x: Tensor, y: Tensor, *, scale: float, dim: int = 1) -> Tensor:
    return torch.exp(pnorm(x, y, p=2, root=False, dim=dim) / scale)


class DistKernel(Protocol):
    def __call__(self, tensor_a: TensorType, tensor_b: TensorType) -> Tensor:
        ...


class KnnOutput(NamedTuple):
    indices: Tensor
    distances: Tensor


@overload
def knn(
    pc: Tensor,
    k: int,
    kernel: DistKernel = ...,
    return_distances: Literal[False] = ...,
    backend: Literal["pykeops", "torch"] = ...,
) -> Tensor:
    ...


@overload
def knn(
    pc: Tensor,
    k: int,
    kernel: DistKernel = ...,
    return_distances: Literal[True] = ...,
    backend: Literal["pykeops", "torch"] = ...,
) -> KnnOutput:
    ...


def knn(
    pc: Tensor,
    k: int,
    kernel: DistKernel = pnorm,
    return_distances: bool = False,
    backend: Literal["pykeops", "torch"] = "torch",
) -> Tensor | KnnOutput:

    G_i = pc[:, None]  # (M**2, 1, 2)
    X_j = pc[None]  # (1, N, 2)
    distances = None

    if backend == "pykeops":
        G_i_lt = LazyTensor(G_i)  # (M**2, 1, 2)
        X_j_lt = LazyTensor(X_j)  # (1, N, 2)
        # symbolic matrix of squared distances
        D_ij = kernel(G_i_lt, X_j_lt)  # (M**2, N)
        # Grid <-> Samples, (M**2, K) integer tensor
        indices = D_ij.argKmin(k, dim=1)  # type: ignore
        if return_distances:
            # Workaround for pykeops currently not supporting differentiation through Kmin
            distances = kernel(pc[:, None], pc[indices, :])
    else:
        D_ij = kernel(G_i, X_j)
        values_indices = D_ij.topk(k, dim=1)
        indices = values_indices.indices
        if return_distances:
            distances = values_indices.values

    if distances is None:
        return indices
    return KnnOutput(indices=indices, distances=distances)
