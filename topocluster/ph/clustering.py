from __future__ import annotations
from typing import NamedTuple, Sequence, cast

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from topocluster.knn import knn, pnorm
from topocluster.ph.utils import plot_persistence

__all__ = ["MergeOutput", "merge_h0", "tomato", "Tomato"]


class MergeOutput(NamedTuple):
    root_idxs: Tensor
    cluster_ids: Tensor
    barcode: Tensor
    tree: Tensor | None = None


def merge_h0(
    neighbor_graph: Tensor | Sequence[Tensor],
    *,
    density_map: Tensor,
    threshold: float,
    store_tree: bool = False
) -> MergeOutput:
    """Merging Data Using Topological Persistence.
    Fast persistence-based merging algorithm specialised for 0-dimensional homology.
    """
    # # Nornalize the density map such that the maximum value is 1
    sort_idxs = density_map.argsort(descending=True)
    # merging happens in a bottom-up fashion, meaning each node defines its own cluster
    root_idxs = torch.arange(len(density_map), dtype=torch.long, device=density_map.device)
    # List of barcodes for each reference index
    barcode_ls: list[Tensor] = []
    # Store the complete time-evolution of the filtration if requested
    tree_ls = [root_idxs.clone()] if store_tree else None
    # Positional indexes for mapping from absolute index to time
    filtration_times = torch.empty_like(sort_idxs)
    

    for t, i in enumerate(sort_idxs[1:], start=1):
        filtration_times[i] = t
        nbr_idxs = neighbor_graph[i]
        # neighbors of v_i with smaller indices (bigger p)
        ls_idxs = nbr_idxs[density_map[nbr_idxs] > density_map[i]]
        # check whether v_i is a local maximum of p
        if len(ls_idxs) > 0:
            # Find all clusters containing nodes in nbd.
            c_max_idx = c_nbd_idxs = root_idxs[ls_idxs]
            if len(c_nbd_idxs) > 1:
                c_max_idx = c_nbd_idxs[density_map[c_nbd_idxs].argmax()]
                # Exclude the local optimum as this will be the merging site.
                c_nbd_idxs = c_nbd_idxs[c_nbd_idxs != c_max_idx]
                # Compute the persistence (death-time - birth-time) of the components.
                persistence = density_map[c_nbd_idxs] - density_map[i]
                merge_mask = persistence < threshold
                #  merge any neighbours below the peristence-threshold, with respect to v_i, into c_max
                if merge_mask.count_nonzero():
                    # Look up the child indexes for each of the lower-star neighbors.
                    child_node_idxs = (root_idxs[:, None] == c_nbd_idxs[merge_mask][None]).nonzero(
                        as_tuple=True
                    )[0]
                    # Merge the lower-star neighbors and their descendents into c_max.
                    root_idxs[child_node_idxs] = c_max_idx

                # record the birth/death times for the connected components
                birth_time = filtration_times[c_nbd_idxs]
                death_time = filtration_times[i].expand(len(c_nbd_idxs))
                barcode_ls.append(torch.stack((birth_time, death_time), dim=-1))

            # assign v_i to cluster c_max
            root_idxs[i] = c_max_idx

        # Extend the stored tree.
        if tree_ls is not None:
            tree_ls.append(root_idxs.clone())

    tree = None if tree_ls is None else torch.stack(tree_ls, -1)
    _, cluster_ids = root_idxs.unique(return_inverse=True)
    barcode = torch.cat(barcode_ls, dim=0)

    return MergeOutput(root_idxs=root_idxs, cluster_ids=cluster_ids, barcode=barcode, tree=tree)


# def merge_h0(
#     neighbor_graph: Tensor | Sequence[Tensor],
#     *,
#     density_map: Tensor,
#     threshold: float,
# ) -> MergeOutput:
#     """Merging Data Using Topological Persistence.
#     Fast persistence-based merging algorithm specialised for 0-dimensional homology.
#     """
#     # Nornalize the density map such that the maximum value is 1
#     # density_map = density_map / density_map.max()
#     sort_idxs = density_map.argsort(descending=False)
#     # merging happens in a bottom-up fashion, meaning each node defines its own cluster
#     root_idxs = torch.arange(len(density_map), dtype=torch.long, device=density_map.device)
#     # List of barcodes for each reference index
#     barcode_ls: list[Tensor] = []
#     # Positional indexes for mapping from absolute index to time
#     filtration_times = torch.empty_like(sort_idxs)

#     for time, ref_idx in enumerate(sort_idxs[1:], start=1):
#         ref_idx = cast(Tensor, ref_idx)
#         filtration_times[ref_idx] = time

#         nbr_idxs = neighbor_graph[ref_idx]
#         # neighbors of v_i with smaller indices (bigger p)
#         ls_idxs = nbr_idxs[density_map[nbr_idxs] > density_map[ref_idx]]
#         # check whether v_i is a local maximum of p
#         if len(ls_idxs) > 0:
#             # all clusters containing nodes in nbd
#             c_max_idx = c_nbd_idxs = root_idxs[ls_idxs]
#             if len(c_nbd_idxs) > 1:
#                 c_max_idx = c_nbd_idxs[density_map[c_nbd_idxs].argmax()]

#                 # Exclude the local optimum
#                 c_nbd_idxs = c_nbd_idxs[c_nbd_idxs != c_max_idx]
#                 # Compute the persistence (death-time - birth-time) of the components
#                 persistence = density_map[c_nbd_idxs] - density_map[ref_idx]
#                 merge_mask = persistence < threshold
#                 #  merge any neighbours below the peristence-threshold, with respect to v_i, into c_max
#                 if merge_mask.count_nonzero():
#                     # Look up the child indexes for each of the lower-star neighbors
#                     child_node_idxs = (root_idxs[:, None] == c_nbd_idxs[merge_mask][None]).nonzero(
#                         as_tuple=True
#                     )[0]
#                     # Merge the lower-star neighbors and their descendents into c_max
#                     root_idxs[child_node_idxs] = c_max_idx

#                 # record the birth/death times for the connected components
#                 # birth_time = filtration_times[c_nbd_idxs]
#                 # death_time = filtration_times[ref_idx].expand(len(c_nbd_idxs))
#                 # barcode_ls.append(torch.stack((birth_time, death_time), dim=-1))

#             # assign v_i to cluster c_max
#             root_idxs[ref_idx] = c_max_idx

#     _, cluster_ids = root_idxs.unique(return_inverse=True)
#     barcode = None
#     # barcode = torch.cat(barcode_ls, dim=0)

#     return MergeOutput(root_idxs=root_idxs, cluster_ids=cluster_ids, barcode=barcode)


def compute_density_map(pc: Tensor, k: int, scale: float) -> tuple[Tensor, Tensor]:
    dists, inds = knn(pc, k=k, kernel="pnorm")
    dists = (-dists / scale).exp().sum(1) / (k * scale)
    return dists / dists.max(), inds


def compute_rips(pc: Tensor, k: int) -> Tensor:
    return knn(pc, k=k, kernel="pnorm")


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
    return merge_h0(neighbor_graph=rips_idxs, density_map=density_map, threshold=threshold)


class Tomato:
    def __init__(
        self, k_kde: int = 100, k_rips: int = 15, scale: float = 0.5, threshold: float = 1.0
    ) -> None:
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.threshold = threshold
        self.barcode: Tensor | None = None

    def plot(self) -> None:
        if self.barcode is not None:
            plot_persistence(self.barcode)
            plt.show()

    def __call__(self, x: Tensor, threshold: float | None = None) -> Tensor:
        threshold = self.threshold if threshold is None else threshold
        output = tomato(
            x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, threshold=threshold
        )

        self.barcode = output.barcode

        return output.cluster_ids
