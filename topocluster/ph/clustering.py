from __future__ import annotations
from typing import NamedTuple, Sequence, cast

import numpy as np
import numpy.typing as npt
import ph_rs
from torch import Tensor

__all__ = [
    "MergeOutput",
    "merge_h0",
    "tomato",
]


class MergeOutput(NamedTuple):
    root_idxs: npt.NDArray[np.uint]
    labels: npt.NDArray[np.uint]


def merge_h0(
    neighbor_graph: Tensor
    | npt.NDArray[np.uint]
    | Sequence[npt.NDArray[np.uint]]
    | Sequence[Sequence[int]],
    *,
    density_map: Tensor | npt.NDArray[np.floating] | Sequence[float],
    threshold: float,
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
    root_idxs = np.array(
        ph_rs.merge_h0(neighbor_graph, density_map=density_map, threshold=threshold)
    )
    _, labels = np.unique(root_idxs, return_inverse=True)

    return MergeOutput(root_idxs=root_idxs, labels=labels)


def tomato(
    neighbor_graph: Tensor
    | npt.NDArray[np.uint]
    | Sequence[npt.NDArray[np.uint]]
    | Sequence[Sequence[int]],
    *,
    density_map: Tensor | npt.NDArray[np.floating],
    threshold: float,
) -> MergeOutput:
    """
    Merges data based on their 0-dimensional persistence according to the ToMATo algorithm.

    :param neighbor_graph: Tensory, array or sequence encoding the neighbourhood of each vertex.
    :param density_map: Tensor, array or sequence encoding the density of each vertex.
    :param threshold: Persistence threshold for merging.

    :returns: Tensor containing the root index (cluster) of each vertex.
    """
    if isinstance(neighbor_graph, Tensor):
        neighbor_graph = cast(np.ndarray, neighbor_graph.detach().cpu().numpy())
    if isinstance(density_map, Tensor):
        density_map = cast(np.ndarray, density_map.detach().cpu().numpy())
    root_idxs = np.array(ph_rs.tomato(neighbor_graph, density_map=density_map, threshold=threshold))
    _, labels = np.unique(root_idxs, return_inverse=True)

    return MergeOutput(root_idxs=root_idxs, labels=labels)
