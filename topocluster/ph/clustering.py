from __future__ import annotations
from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from topocluster import search
from topocluster.ph.utils import plot_persistence

__all__ = ["MergeOutput", "merge_h0", "tomato", "Tomato"]


class MergeOutput(NamedTuple):
    root_idxs: Tensor
    labels: Tensor
    barcode: Tensor
    tree: Tensor | None = None


def merge_h0(
    neighbor_graph: Tensor | Sequence[Tensor],
    *,
    density_map: Tensor,
    threshold: float,
    store_tree: bool = False,
) -> MergeOutput:
    """Merging Data Using Topological Persistence.
    Fast persistence-based merging algorithm specialised for 0-dimensional homology.
    """
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
        # neighbors of v_i.
        nbr_idxs = neighbor_graph[i]
        # neighbors of v_i with smaller indices (bigger p).
        us_idxs = nbr_idxs[density_map[nbr_idxs] > density_map[i]]
        # check whether v_i is a local maximum of p
        if len(us_idxs) > 0:
            # Find all clusters containing nodes in nbd.
            c_max_idx = c_nbd_idxs = root_idxs[us_idxs]
            p_vi = density_map[i]
            if len(c_nbd_idxs) > 1:
                c_max_idx = c_nbd_idxs[density_map[c_nbd_idxs].argmax()]
                # Exclude the local optimum as this will be the merging site.
                c_nbd_idxs = c_nbd_idxs[c_nbd_idxs != c_max_idx]
                # Compute the persistence (death-time - birth-time) of the components.
                p_c_nbd = density_map[c_nbd_idxs]
                persistence = p_c_nbd - p_vi
                #  merge any neighbours below the peristence-threshold into c_max
                merge_mask = persistence < threshold
                num_merges = int(merge_mask.count_nonzero())
                if num_merges:
                    # Look up the child indexes for each of the upper-star neighbors.
                    child_node_idxs = (root_idxs[:, None] == c_nbd_idxs[merge_mask][None]).nonzero(
                        as_tuple=True
                    )[0]
                    # Merge the upper-star neighbors and their descendents into c_max.
                    root_idxs[child_node_idxs] = c_max_idx

                    # record the birth/death times for the connected components
                    barcode_ls.append(
                        torch.stack((p_c_nbd[merge_mask], p_vi.expand(num_merges)), dim=-1)
                    )
            # assign v_i to cluster c_max
            root_idxs[i] = c_max_idx

        # Extend the merging tree.
        if tree_ls is not None:
            tree_ls.append(root_idxs.clone())

    tree = None if tree_ls is None else torch.stack(tree_ls, -1)
    cluster_idxs, labels = root_idxs.unique(return_inverse=True)

    p_C = density_map[cluster_idxs]
    p_min = density_map[sort_idxs[-1]].item()
    # Record the birth-death pairs for the modes of the data
    barcode_ls.append(torch.stack([p_C, torch.full_like(p_C, p_min)], dim=-1))
    barcode = torch.cat(barcode_ls, dim=0).long()

    return MergeOutput(root_idxs=root_idxs, labels=labels, barcode=barcode, tree=tree)


def compute_density_map(pc: Tensor, k: int, scale: float) -> tuple[Tensor, Tensor]:
    knn = search.KnnExact(k=k, kernel="pnorm")
    out = knn(pc, return_distances=True)
    dists = (-out.distances / scale).exp().sum(1) / (k * scale)
    return dists / dists.max(), out.indices


def compute_rips(pc: Tensor, k: int) -> Tensor:
    knn = search.KnnExact(k=k, kernel="pnorm")
    return knn(pc, return_distances=False)


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

        return output.labels
