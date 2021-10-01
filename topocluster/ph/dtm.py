from __future__ import annotations

import torch
from torch import Tensor

from topocluster.knn import DistKernel, knn, pnorm

__all__ = ["distance_to_measure", "dtm_density"]


def distance_to_measure(pc: Tensor, *, k: int, q: int = 2, kernel: DistKernel = pnorm) -> Tensor:
    """
    Computes the distance to the empirical measure defined by a point set.

    :param k: Number of neighbors (possibly including the point itself).
    :param q: Order used to compute the distance to measure.
    :param kernel: Kernel used to compute the pairwise distances for k-nn search.
    """
    distances = knn(pc, k=k, return_distances=True, kernel=kernel).distances
    return distances.mean(-1) ** (1.0 / q)


def dtm_density(
    pc: Tensor,
    *,
    k: int,
    q: int = 2,
    kernel: DistKernel = pnorm,
    normalize: bool = False,
    dim: int | None = None,
) -> Tensor:
    """
    Estimate the density based on the distance to the empirical measure defined by a point set.

    :param q: Order used to compute the distance to measure.
    :param kernel: Kernel used to compute the pairwise distances for k-nn search.
    :param normalize: Normalize the density so it corresponds to a probability measure on ℝᵈ.
        Only available for the Euclidean metric, defaults to False.

    .. note::
        When the dimension is high, using it as an exponent can quickly lead to under- or overflows.
        We recommend using a small fixed value instead in those cases, even if it won't have the
        same nice theoretical properties as the dimension.
    """
    dim = pc.size(1) if dim is None else dim
    distances = knn(pc, k=k, return_distances=True, kernel=kernel).distances ** q
    dtm = distances.sum(-1)
    if normalize:
        dtm /= (torch.arange(1, k + 1) ** (q / dim)).sum()
    density = dtm ** (-dim / q)
    n_samples = len(pc)
    if normalize:
        import math

        # Volume of d-ball
        v_d = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
        density /= n_samples * v_d
    return density
