from __future__ import annotations
from itertools import groupby
from typing import NamedTuple, Sequence, cast

import numpy as np
import numpy.typing as npt
import ph_rs
import torch
from torch import Tensor

__all__ = ["MergeOutput", "cluster_h0", "sort_persistence_pairs"]


class MergeOutput(NamedTuple):
    root_idxs: npt.NDArray[np.uint]
    labels: npt.NDArray[np.uint]
    persistence_pairs: list[tuple[int, int | None]]


def cluster_h0(
    neighbor_graph: Tensor
    | npt.NDArray[np.uint]
    | Sequence[npt.NDArray[np.uint]]
    | Sequence[Sequence[int]],
    *,
    density_map: Tensor | npt.NDArray[np.floating] | Sequence[float],
    threshold: float,
    greedy: bool = False,
) -> MergeOutput:
    """
    Merges data based on their 0-dimensional persistence.

    :param neighbor_graph: Tensory, array or sequence encoding the neighbourhood of each vertex.
    :param density_map: Tensor, array or sequence encoding the density of each vertex.
    :param threshold: Persistence threshold for merging.

    :returns: Tensor containing the root index (cluster) of each vertex.
    """
    if isinstance(neighbor_graph, Tensor):
        neighbor_graph = cast(np.ndarray, neighbor_graph.detach().cpu().numpy())
    if isinstance(density_map, Tensor):
        density_map = cast(np.ndarray, density_map.detach().cpu().numpy())
    root_idxs_ls, persistence_pairs = ph_rs.cluster_h0(
        neighbor_graph, density_map=density_map, threshold=threshold, greedy=greedy
    )
    root_idxs = np.array(root_idxs_ls)
    _, labels = np.unique(root_idxs, return_inverse=True)

    return MergeOutput(root_idxs=root_idxs, labels=labels, persistence_pairs=persistence_pairs)


class SortedPersistencePairs(NamedTuple):
    persistence_pairs: Tensor
    persistence: Tensor
    inf_components: Tensor

    def __len__(self) -> int:
        return len(self.persistence_pairs) + len(self.inf_components)


def sort_persistence_pairs(
    persistence_pairs: list[tuple[int, int | None]], density_map: Tensor
) -> SortedPersistencePairs:
    pers_pairs_fin_ls, pers_pairs_inf_ls = (
        list(group[1]) for group in groupby(persistence_pairs, lambda x: x[1] is None)
    )
    pers_pairs_inf_ls = [pair[0] for pair in pers_pairs_inf_ls]
    pers_pairs_fin = torch.as_tensor(pers_pairs_fin_ls, dtype=torch.long)
    pers_pairs_inf = torch.as_tensor(pers_pairs_inf_ls, dtype=torch.long)

    pers_fin = -torch.diff(density_map[pers_pairs_fin], dim=1).squeeze(1)
    pers_fin_sorted, sort_idxs = pers_fin.sort(descending=True)
    pers_pairs_fin_sorted = pers_pairs_fin[sort_idxs]

    return SortedPersistencePairs(
        persistence_pairs=pers_pairs_fin_sorted,
        persistence=pers_fin_sorted,
        inf_components=pers_pairs_inf,
    )
