from __future__ import annotations
import numpy as np
from typing import NamedTuple, Sequence

import numpy.typing as npt
import torch
from torch import Tensor

from topocluster import search

__all__ = ["MergeOutput", "merge_h0", "tomato", "Tomato"]


class MergeOutput(NamedTuple):
    root_idxs: Tensor
    labels: Tensor


def merge_h0(
    neighbor_graph: Tensor | npt.NDArray[np.uint],
    *,
    density_map: Tensor | npt.NDArray[np.floating],
    threshold: float,
) -> MergeOutput:
    """Merging Data Using Topological Persistence.
    Fast persistence-based merging algorithm specialised for 0-dimensional homology.
    """
    import ph_rs

    if isinstance(neighbor_graph, Tensor):
        neighbor_graph = neighbor_graph.detach().cpu().numpy()
    if isinstance(density_map, Tensor):
        density_map = density_map.detach().cpu().numpy()
    root_idxs = torch.as_tensor(ph_rs.merge_h0(neighbor_graph, density_map, threshold))
    _, labels = root_idxs.unique(return_inverse=True)

    return MergeOutput(root_idxs=root_idxs, labels=labels)


def merge_h0_torch(
    neighbor_graph: Tensor | Sequence[Tensor],
    *,
    density_map: Tensor,
    threshold: float,
) -> MergeOutput:
    """Merging Data Using Topological Persistence.
    Fast persistence-based merging algorithm specialised for 0-dimensional homology.
    """
    sort_idxs = density_map.argsort(descending=True)
    # merging happens in a bottom-up fashion, meaning each node defines its own cluster
    root_idxs = torch.arange(len(density_map), dtype=torch.long, device=density_map.device)
    # List of barcodes for each reference index
    # Store the complete time-evolution of the filtration if requested
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

            # assign v_i to cluster c_max
            root_idxs[i] = c_max_idx

    _, labels = root_idxs.unique(return_inverse=True)

    return MergeOutput(root_idxs=root_idxs, labels=labels)


def compute_density_map(pc: Tensor, k: int, scale: float) -> tuple[Tensor, Tensor]:
    knn = search.KnnExact(k=k, p=2)
    out = knn(pc, return_distances=True)
    dists = (-out.distances / scale).exp().sum(1) / (k * scale)
    return dists / dists.max(), out.indices


def compute_rips(pc: Tensor, k: int) -> Tensor:
    knn = search.KnnExact(k=k, p=2)
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

    def __call__(self, x: Tensor, threshold: float | None = None) -> Tensor:
        threshold = self.threshold if threshold is None else threshold
        output = tomato(
            x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, threshold=threshold
        )

        return output.labels
