from typing import Any, Callable, Tuple, Type, Union

import torch
from torch import Tensor, jit
import torch.distributions as td
from torch.nn import functional as F

__all__ = [
    "RoundSTE",
    "compute_density_map",
    "compute_rips",
    "dot_product",
    "knn",
    "logit",
    "normalized_softmax",
    "pairwise_L2sqr",
    "sample_concrete",
    "sum_except_batch",
    "to_discrete",
]


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor) -> Tensor:
        return inputs.round()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        """Straight-through estimator"""
        return grad_output


def to_discrete(inputs: Tensor, dim: int = 1) -> Tensor:
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    else:
        argmax = inputs.argmax(dim=1)
        return F.one_hot(argmax, num_classes=inputs.size(1))


def sample_concrete(logits: Tensor, temperature: float) -> Tensor:
    """Sample from the concrete/gumbel softmax distribution for
    differentiable discretization.
    Args:
        logits (Tensor): Logits to be transformed.
        temperature (float): Temperature of the distribution. The lower
        the temperature the closer the distribution comes to approximating
        a discrete distribution.
    Returns:
        Tensor: Samples from a concrete distribution with the
        given temperature.
    """
    Concrete: Union[Type[td.RelaxedBernoulli], Type[td.RelaxedOneHotCategorical]]
    if logits.dim() <= 1 or logits.size(1) <= 1:
        Concrete = td.RelaxedBernoulli
    else:
        Concrete = td.RelaxedOneHotCategorical
    concrete = Concrete(logits=logits, temperature=temperature)
    return concrete.rsample()


def logit(p: Tensor, eps: float = 1e-8) -> Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def sum_except_batch(x: Tensor, keepdim: bool = False) -> Tensor:
    return x.flatten(start_dim=1).sum(-1, keepdim=keepdim)


def dot_product(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    return torch.sum(x * y, dim=-1, keepdim=keepdim)


@jit.script
def normalized_softmax(logits: Tensor) -> Tensor:
    max_logits, _ = logits.max(dim=1, keepdim=True)
    unnormalized = torch.exp(logits - max_logits)
    return unnormalized / unnormalized.norm(p=2, dim=-1, keepdim=True)


def pairwise_L2sqr(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return (tensor_a - tensor_b) ** 2


def rbf(x: Tensor, y: Tensor, scale: float) -> Tensor:
    return torch.exp(-torch.norm(x - y, axis=1) ** 2 / scale)


def compute_density_map(pc: Tensor, k: int, scale: float) -> Tuple[Tensor, Tensor]:
    dists, inds = knn(pc, k=k, kernel=pairwise_L2sqr)
    dists = torch.sum(torch.exp(-dists / scale), dim=1) / (k * scale)
    return dists / dists.max(), inds


def compute_rips(pc: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    return knn(pc=pc, k=k, kernel=pairwise_L2sqr)


def knn(
    pc: Tensor, k: int, kernel: Callable[[Tensor, Tensor], Tensor] = pairwise_L2sqr
) -> Tuple[Tensor, Tensor]:
    X_i = pc[:, None, :]
    X_j = pc[None, :, :]  # (1, N, 2)
    D_ij = kernel(X_i, X_j).sum(-1)  # (M**2, N) symbolic matrix of distances
    indKNN = D_ij.topk(k=k, dim=1, largest=False)[1]
    # indKNN = soft_rank(D_ij, direction="ASCENDING", regularization_strength=0.01)
    return D_ij, indKNN
