from __future__ import annotations
from typing import Callable

import torch
from torch import Tensor

__all__ = [
    "pairwise_l2sqr",
    "knn",
    "rbf",
]


def pairwise_l2sqr(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return (tensor_a - tensor_b) ** 2


def rbf(x: Tensor, y: Tensor, scale: float, dim: int = 1) -> Tensor:
    return torch.exp(-torch.norm(x - y, dim=dim) ** 2 / scale)


def knn(
    pc: Tensor, k: int, kernel: Callable[[Tensor, Tensor], Tensor] = pairwise_l2sqr
) -> tuple[Tensor, Tensor]:
    from pykeops.torch import LazyTensor

    G_i = LazyTensor(pc[:, None, :])  # (M**2, 1, 2)
    X_j = LazyTensor(pc[None, :, :])  # (1, N, 2)
    D_ij = kernel(G_i, X_j).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(k, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    # Workaround for pykeops currently not supporting differentiation through Kmin
    to_nn_alt = kernel(pc[:, None], pc[indKNN, :]).sum(-1)

    return to_nn_alt, indKNN
