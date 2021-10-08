from __future__ import annotations

import torch
from torch import Tensor

from topocluster.search import Knn

__all__ = ["DTM", "DTMDensity"]


class DTM:
    @staticmethod
    def from_dists(dists: Tensor, q: int = 2) -> Tensor:
        """
        Computes the distance to the empirical measure defined by a point set.

        :param dists: Precomputed k-nearest-neighbor distances.
        :param q: Order used to compute the distance to measure.
        """
        return dists.mean(-1) ** (1.0 / q)

    @staticmethod
    def with_knn(
        x: Tensor,
        *,
        knn: Knn,
        q: int = 2,
    ) -> Tensor:
        """
        Computes the distance to the empirical measure defined by a point set.

        :param x: Point set to compute the empirical DTM measure for.
        :param knn: k-NN searcher to use for computing the distances that serve as the basis
            of the density estimate.
        :param q: Order used to compute the distance to measure.
        :param normalize: Whether to normalize the vectors for the k-nn search.
        """
        distances = knn(x, return_distances=True).distances
        return DTM.from_dists(distances, q=q)


class DTMDensity:
    @staticmethod
    def from_dists(
        dists: Tensor,
        *,
        dim: int,
        q: float | None = None,
        normalize: bool = False,
        weights: Tensor | None = None,
    ) -> Tensor:
        """
        Estimate the density based on the distance to the empirical measure defined by a point set.

        :param dists: Precomputed k-nearest-neighbor distances.
        :param dim: Final exponent representing the dimension. Defaults to the dimension.
        :param q: Order used to compute the distance to measure; defaults to dim.
        :param normalize: Normalize the density so it corresponds to a probability measure on ℝᵈ.
            Only available for the Euclidean metric, defaults to False.

        .. note::
            When the dimension is high, using it as an exponent can quickly lead to under- or overflows.
            We recommend using a small fixed value instead in those cases, even if it won't have the
            same nice theoretical properties as the dimension.

        :param weights: weights of each of the k neighbors; should sum to 1.
        """
        k = dists.size(1)
        if weights is None:
            weights = dists.new_full((k,), 1 / k)
        if q is None:
            q = dim

        dtm = ((dists ** q) * weights).sum(-1)
        if normalize:
            dtm /= (torch.arange(1, k + 1, device=dists.device) ** (q / dim) * weights).sum()
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
        x: Tensor,
        *,
        knn: Knn,
        q: float | None = None,
        normalize: bool = False,
        dim: int | None = None,
    ) -> Tensor:
        """
        Estimate the density based on the distance to the empirical measure defined by a point set.

        :param x: Point set from which to build the empirical density estimate.
        :param q: Order used to compute the distance to measure.
        :param knn: k-NN searcher to use for computing the distances that serve as the basis
            of the density estimate.
        :param normalize: Normalize the density so it corresponds to a probability measure
            on ℝᵈ.

        .. note::
            When the dimension is high, using it as an exponent can quickly lead to under- or overflows.
            We recommend using a small fixed value instead in those cases, even if it won't have the
            same nice theoretical properties as the dimension.

        :param dim: Final exponent representing the dimension. Defaults to the dimension.
        """
        dim = x.size(1) if dim is None else dim
        distances = knn(x, return_distances=True).distances

        return DTMDensity.from_dists(dists=distances, q=q, dim=dim, normalize=normalize)
