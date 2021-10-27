from __future__ import annotations
from typing import Sequence

import numpy as np
import numpy.typing as npt


def cluster_h0(
    neighbor_graph: npt.NDArray[np.uint] | Sequence[npt.NDArray[np.uint]] | Sequence[Sequence[int]],
    *,
    density_map: npt.NDArray[np.floating] | Sequence[float],
    threshold: float,
    greedy: bool,
) -> tuple[list[int], list[tuple[int, int | None]]]:
    """
    Merges data based on their 0-dimensional persistence.

    :param neighbor_graph: Array or sequence encoding the neighbourhood of each vertex.
    :param density_map: Array or sequence encoding the density of each vertex.
    :param threshold: Persistence threshold for merging.
    :param greedy: Whether to make cluster assignments greedily (based on the maximumdensity of the
        root indexes instead of the maximum density of the neighbour indexes).

    :returns: Sequence containing the root index (cluster) of each vertex along with a Sequence of
        persistence pairs.
    """
