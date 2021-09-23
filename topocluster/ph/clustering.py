from __future__ import annotations
from typing import NamedTuple, Sequence, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from topocluster.ph.utils import plot_persistence
from topocluster.utils.torch_ops import knn, pairwise_l2sqr

__all__ = ["MergeOutput", "zero_dim_merge", "tomato", "Tomato"]


class MergeOutput(NamedTuple):
    root_idxs: Tensor
    cluster_ids: Tensor
    persistence_pairs: Tensor


def zero_dim_merge(
    neighbor_graph: Tensor | Sequence[Tensor],
    *,
    density_map: Tensor,
    threshold: float,
) -> MergeOutput:
    """Merging Data Using Topological Persistence.
    Fast persistence-based merging algorithm specialised for 0-dimensional homology.
    """
    sort_idxs = density_map.argsort(descending=True)
    # precompose the pairwise inequalities
    pairwise_lt = density_map[None] < density_map[:, None]
    # merging happens in a bottom-up fashion, meaning each node defines its own cluster
    root_idxs = torch.arange(len(density_map), dtype=torch.long, device=density_map.device)
    # List of barcodes for each reference index
    persistence_pairs_ls: list[Tensor] = []
    # Positional indexes for mapping from absolute index to rank
    ranks = torch.empty_like(sort_idxs)

    for rank, ref_idx in enumerate(sort_idxs[1:], start=1):
        ref_idx = cast(Tensor, ref_idx)
        ranks[ref_idx] = rank
        nbr_idxs = neighbor_graph[ref_idx]
        # neighbors of v_i with smaller indices (bigger p)
        ls_idxs = nbr_idxs[pairwise_lt[nbr_idxs, ref_idx]]
        # check whether v_i is a local maximum of p
        if len(ls_idxs) > 0:
            # all clusters containing nodes in nbd
            c_max_idx = c_nbd_idxs = root_idxs[ls_idxs]
            if len(c_nbd_idxs) > 1:
                c_max_idx = c_nbd_idxs[density_map[c_nbd_idxs].argmax()]
                c_nbd_idxs = c_nbd_idxs[c_nbd_idxs != c_max_idx]
                persistence = density_map[c_nbd_idxs] - density_map[ref_idx]
                merge_mask = persistence < threshold
                #  merge any neighbours below the peristence-threshold, with respeft to v_i, into c_max
                if merge_mask.count_nonzero():
                    child_node_idxs = (root_idxs[:, None] == c_nbd_idxs[merge_mask][None]).nonzero(
                        as_tuple=True
                    )[0]
                    root_idxs[child_node_idxs] = c_max_idx

                birth_time = ranks[c_nbd_idxs]
                death_time = ranks[ref_idx].expand(len(c_nbd_idxs))
                persistence_pairs_ls.append(torch.stack((birth_time, death_time), dim=-1))
            # assign v_i to cluster c_max
            root_idxs[ref_idx] = c_max_idx

    _, cluster_ids = root_idxs.unique(return_inverse=True)
    persistence_pairs = torch.cat(persistence_pairs_ls, dim=0)

    return MergeOutput(
        root_idxs=root_idxs, cluster_ids=cluster_ids, persistence_pairs=persistence_pairs
    )


def compute_density_map(pc: Tensor, k: int, scale: float) -> tuple[Tensor, Tensor]:
    dists, inds = knn(pc, k=k, kernel=pairwise_l2sqr)
    dists = (-dists / scale).exp().sum(1) / (k * scale)
    return dists / dists.max(), inds


def compute_rips(pc: Tensor, k: int) -> Tensor:
    return knn(pc=pc, k=k, kernel=pairwise_l2sqr)[1]


def tomato(pc: Tensor, k_kde: int, k_rips: int, scale: float, threshold: float) -> MergeOutput:
    """Topological mode analysis tool (Chazal et al., 2013).

    :param pc: The point-cloud to be clustered
    :param k_kde: Number of neighbors to use when computing the density map.
    :param k_rips: Number of neighbors to use when computing the Rips graph.
    :param scale: Bandwidth of the kernel used when computing the density map.
    threshold (float): Thresholding parameter (tau in the paper)

    :returns: Named tuple containing the root idxs, corresponding cluster ids, and barcodes.
    """
    density_map, _ = compute_density_map(pc, k_kde, scale)
    _, rips_idxs = compute_rips(pc, k=k_rips)
    return zero_dim_merge(neighbor_graph=rips_idxs, density_map=density_map, threshold=threshold)


class Tomato:
    def __init__(
        self, k_kde: int = 100, k_rips: int = 15, scale: float = 0.5, threshold: float = 1.0
    ) -> None:
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.threshold = threshold
        self.persistence_pairs: Tensor | None = None

    def plot(self) -> None:
        if self.persistence_pairs is not None:
            plot_persistence(self.persistence_pairs)
            plt.show()

    def __call__(self, x: Tensor, threshold: float | None = None) -> Tensor:
        threshold = self.threshold if threshold is None else threshold
        output = tomato(
            x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, threshold=threshold
        )

        self.persistence_pairs = output.persistence_pairs

        return output.cluster_ids
