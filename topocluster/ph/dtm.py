from __future__ import annotations

import torch
from torch import Tensor

from topocluster.knn import DistKernel, knn, pnorm

__all__ = ["DTM", "DTMDensity"]


class DTM:
    @staticmethod
    def from_dists(dists: Tensor, q: int = 2) -> Tensor:
        """
        Computes the distance to the empirical measure defined by a point set.

        :param q: Order used to compute the distance to measure.
        """
        return dists.mean(-1) ** (1.0 / q)

    @staticmethod
    def with_knn(pc: Tensor, *, k: int, q: int = 2, kernel: DistKernel = pnorm) -> Tensor:
        """
        Computes the distance to the empirical measure defined by a point set.

        :param k: Number of neighbors (possibly including the point itself).
        :param q: Order used to compute the distance to measure.
        :param kernel: Kernel used to compute the pairwise distances for k-nn search.
        """
        distances = knn(pc, k=k, return_distances=True, kernel=kernel).distances
        return DTM.from_dists(distances, q=q)


class DTMDensity:
    @staticmethod
    def from_dists(
        dists: Tensor,
        *,
        dim: int,
        q: int = 2,
        normalize: bool = False,
    ) -> Tensor:
        """
        Estimate the density based on the distance to the empirical measure defined by a point set.

        :param k: Number of neighbors (possibly including the point itself).
        :param q: Order used to compute the distance to measure.
        :param kernel: Kernel used to compute the pairwise distances for k-nn search.
        :param normalize: Normalize the density so it corresponds to a probability measure on ℝᵈ.
            Only available for the Euclidean metric, defaults to False.

        .. note::
            When the dimension is high, using it as an exponent can quickly lead to under- or overflows.
            We recommend using a small fixed value instead in those cases, even if it won't have the
            same nice theoretical properties as the dimension.

        :param dim: Final exponent representing the dimension. Defaults to the dimension.
        """
        k = dists.size(1)
        dtm = dists.sum(-1)
        if normalize:
            dtm /= (torch.arange(1, k + 1) ** (q / dim)).sum()
        density = dtm ** (-dim / q)
        n_samples = len(dists)
        if normalize:
            import math

            # Volume of d-ball
            v_d = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
            density /= n_samples * v_d
        return density

    @staticmethod
    def with_knn(
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

        :param k: Number of neighbors (possibly including the point itself).
        :param q: Order used to compute the distance to measure.
        :param kernel: Kernel used to compute the pairwise distances for k-nn search.
        :param normalize: Normalize the density so it corresponds to a probability measure on ℝᵈ.
            Only available for the Euclidean metric, defaults to False.

        .. note::
            When the dimension is high, using it as an exponent can quickly lead to under- or overflows.
            We recommend using a small fixed value instead in those cases, even if it won't have the
            same nice theoretical properties as the dimension.

        :param dim: Final exponent representing the dimension. Defaults to the dimension.
        """
        dim = pc.size(1) if dim is None else dim
        distances = knn(pc, k=k, return_distances=True, kernel=kernel).distances ** q
        return DTMDensity.from_dists(dists=distances, q=q, dim=dim, normalize=normalize)
