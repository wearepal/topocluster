from typing import Sequence

import numpy as np
import numpy.typing as npt


def merge_h0(
    neighbor_graph: npt.NDArray[np.uint] | Sequence[npt.NDArray[np.uint]] | Sequence[Sequence[int]],
    *,
    density_map: npt.NDArray[np.floating] | Sequence[float],
    threshold: float,
) -> list[int]:
    """
    Merges data based on their 0-dimensional persistence.

    :param neighbor_graph: Array or sequence encoding the neighbourhood of each vertex.
    :param density_map: Array or sequence encoding the density of each vertex.
    :param threshold: Persistence threshold for merging.

    :returns: Sequence containing the root index (cluster) of each vertex.
    """
