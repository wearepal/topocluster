from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import umap

from topocluster.utils.numpy_ops import compute_density_map, compute_rips


class Tomato(nn.Module):
    def __init__(
        self,
        k_kde: int = 100,
        k_rips: int = 15,
        scale: float = 0.5,
        batch_size: Optional[int] = None,
        umap_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.batch_size = batch_size
        self._set_umap_defaults(umap_kwargs)
        self.reducer = umap.UMAP(**umap_kwargs) if umap_kwargs is not None else None
        self._labels: Tensor
        self._pers_pairs: Tensor

    def _set_umap_defaults(self, umap_kwargs: Optional[Dict[str, Any]]) -> None:
        if umap_kwargs is not None:
            umap_kwargs.setdefault("n_components", 10)
            umap_kwargs.setdefault("n_neighbors", self.k_rips)
            umap_kwargs.setdefault("random_state", 42)

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(self._pers_pairs[:, 1], self._pers_pairs[:, 0], s=15, c="orange")  # type: ignore[arg-type]
        ax.plot(np.array([0, 1]), np.array([0, 1]), c="black", alpha=0.6)  # type: ignore[call-arg]
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("Persistence Diagram")

        return fig

    def fit(self, x: Union["np.ndarray[np.float32]", Tensor], threshold: float = 1) -> "Tomato":
        if isinstance(x, Tensor):
            x = x.cpu().detach().numpy()
        assert isinstance(x, np.ndarray)
        x = x.reshape(num_samples := x.shape[0], -1)
        #  Reduce the dimensionality of the data first with UMAP
        if self.reducer is not None:
            x = self.reducer.fit_transform(x)
        batch_size = self.batch_size or num_samples

        def _partialled(
            _x: np.ndarray,
        ) -> Tuple[Dict[int, "np.ndarray[np.int64]"], "np.ndarray[np.float32]"]:
            return tomato(
                _x,
                self.k_kde,
                self.k_rips,
                self.scale,
                threshold,
            )

        clusters: Mapping[int, Union[List[np.int64], "np.array[np.int64]"]]
        pers_pairs: "np.ndarray[np.float32]"

        if batch_size < num_samples:
            clusters = defaultdict(list)
            pers_pairs_ls: List["np.array[np.float32]"] = []
            batches = np.array_split(x, indices_or_sections=batch_size, axis=0)
            for batch in batches:
                clusters_b, pers_pairs_b = _partialled(batch)
                for key, values in clusters_b.items():
                    clusters[key].extend(values)
                pers_pairs_ls.append(pers_pairs_b)

            pers_pairs = np.concatenate(pers_pairs_ls, axis=0)
        else:
            clusters, pers_pairs = _partialled(x)

        cluster_labels = np.empty(x.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k

        cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.int32)
        pers_pairs = torch.as_tensor(pers_pairs, dtype=torch.float32)

        self._labels = cluster_labels
        self._pers_pairs = pers_pairs

        return self

    def fit_transform(
        self, x: Union["np.ndarray[np.float32]", Tensor], threshold: float = 1
    ) -> Tensor:
        self.fit(x=x, threshold=threshold)
        return self._labels


def tomato(
    pc: np.ndarray, k_kde: int, k_rips: int, scale: float, threshold: float
) -> Tuple[Dict[int, "np.ndarray[np.int32]"], "np.ndarray[np.float32]"]:
    """Topological mode analysis tool (Chazal et al., 2013).

    Args:
        pc (np.ndarray): The point-cloud to be clustered
        k_kde (int): Number of neighbors to use when computing the density map
        k_rips (int): Number of neighbors to use when computing the Rips graph.
        scale (float): Bandwidth of the kernel used when computing the density map.
        threshold (float): Thresholding parameter (tau in the paper)

    Returns:
        Tuple[Dict[int, np.ndarray[np.int]], np.ndarray[np.float32]]: Clusters and their pers_pairs
    """
    pc = pc.astype(float)
    #  Compute the k-NN KDE
    density_map, _ = compute_density_map(pc, k_kde, scale)
    density_map = density_map.astype(np.float32)
    sorted_idxs = np.argsort(density_map)
    density_map_sorted = density_map[sorted_idxs]
    pc = pc[sorted_idxs]

    _, rips_idxs = compute_rips(pc, k=k_rips)
    entries, pers_pairs = cluster(density_map_sorted, rips_idxs, threshold=threshold)
    if threshold == 1:
        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])
        result = []
        if see.size > 0:
            for i in np.unique(see[:, 0]):
                result.append(
                    [
                        see[np.where(see[:, 0] == i)[0]][0, 0],
                        max(see[np.where(see[:, 0] == i)[0]][:, 1]),
                    ]
                )
            result = np.array(result)
            for key, value in entries.items():
                entries[key] = sorted_idxs[value]
        else:
            warnings.warn("Clustering unsuccessful; consider expanding the VRC neighbourhood.")
        return entries, density_map_sorted[result]
    else:
        for key, value in entries.items():
            entries[key] = sorted_idxs[value]
        return entries, np.array([[0, 0]])


@jit(nopython=True)
def find_entry_idx_by_point(entries: Dict[int, List[int]], point_idx: int) -> np.int64:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return np.int64(index)
    return np.int64(point_idx)


@jit(nopython=True)
def cluster(
    density_map: np.ndarray, rips_idxs: np.ndarray, threshold: float
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    N = density_map.shape[0]
    pers_pairs = np.array([[-1, -1]])
    #  initialize the union-find data-structure with the final index pointing only to itself

    entries = {N - 1: np.array([N - 1])}

    for i in np.arange(N - 2, -1, -1):
        nbr_idxs = rips_idxs[i]
        # compute the upper star Si = {(i, j1), · · · , (i, jk)} of vertex i in R_δ(L);
        us_idxs = nbr_idxs[nbr_idxs > i]
        # check whether vertex i is a local maximum of f within R_δ
        if us_idxs.size == 0:
            entries[i] = np.array([i])  #  create an entry for the local maximum
        else:
            # approximate the gradient of the underlying probability density function by connecting
            # i to its neighbour in the graph with the highest function value
            g_i = np.max(us_idxs)  #  find the maximum index
            # Attach vertex i to the tree t containing g(i)
            e_up = find_entry_idx_by_point(entries, g_i)
            entries[e_up] = np.append(entries[e_up], i)
            entries, pers_pairs_i = merge(
                density_map=density_map,
                entries=entries,
                ref_idx=i,
                e_up=e_up,
                us_idxs=us_idxs,
                threshold=threshold,
            )
            if len(pers_pairs_i) > 1:
                pers_pairs = np.append(pers_pairs, pers_pairs_i, axis=0)

    return entries, pers_pairs


@jit(nopython=True)
def merge(
    density_map: np.ndarray,
    entries: Dict[int, List[int]],
    ref_idx: int,
    e_up: int,
    us_idxs: List[int],
    threshold: float,
) -> Tuple[Dict[int, List[int]], "np.ndarray[np.int64]"]:
    pers_pairs = np.array([[-1, -1]])

    for idx in us_idxs:
        entry_idx = find_entry_idx_by_point(entries, idx)

        if e_up != entry_idx:
            persistence = density_map[entry_idx] - density_map[ref_idx]

            if persistence < threshold:
                entries[e_up] = np.append(entries[e_up], entries[entry_idx])
                entries.pop(entry_idx)

            pers_pairs = np.append(pers_pairs, np.array([[entry_idx, ref_idx]]), axis=0)

    return entries, pers_pairs


# # @jit(nopython=True)
# def cluster2(
#     density_map: np.ndarray, rips_idxs: np.ndarray, threshold: float
# ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
#     pers_pairs = np.array([[-1, -1]])
#     sort_idxs = density_map.argsort()
#     density_map = density_map[sort_idxs]
#     #  initialize the union-find data-structure with the final index pointing only to itself
#     N = density_map.shape[0]
#     entries = np.full(N, dtype=np.int64, fill_value=np.nan)
#     entries[N - 1] = N - 1

#     for i in np.arange(N - 2, -1, -1):
#         nbr_idxs = rips_idxs[i]
#         # compute the upper star Si = {(i, j1), · · · , (i, jk)} of vertex i in R_δ(L);
#         us_idxs = nbr_idxs[nbr_idxs > i]
#         # check whether vertex i is a local maximum of f within R_δ
#         if us_idxs.size == 0:
#             entries[i] = i
#         else:
#             # approximate the gradient of the underlying probability density function by connecting
#             # i to its neighbour in the graph with the highest function value
#             g_i = np.max(us_idxs)  #  find the maximum index in the neighbourhood of the vertex
#             # Attach vertex i to the tree t containing g(i)
#             e_up = entries[g_i]
#             entries[i] = e_up

#             entries, pers_pairs_i = merge2(
#                 density_map=density_map,
#                 entries=entries,
#                 ref_idx=i,
#                 e_up=e_up,
#                 us_idxs=us_idxs,
#                 threshold=threshold,
#             )
#             if len(pers_pairs_i) > 1:
#                 pers_pairs = np.append(pers_pairs, pers_pairs_i, axis=0)

#     return entries, pers_pairs


# # @jit(nopython=True)
# def merge2(
#     density_map: np.ndarray[np.float32],
#     entries: np.ndarray[np.int64],
#     ref_idx: int,
#     e_up: int,
#     us_idxs: List[int],
#     threshold: float,
# ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
#     pers_pairs = [[-1, -1]]
#     # Find the root associated with the local maximum
#     entry_idxs = entries[us_idxs]
#     # Compute the persistence of each neighbourhood-vertex with respect to the reference index
#     persistence = density_map[entry_idxs] - density_map[ref_idx]
#     # Merge those vertices with persistence lower than the threshold into the root, e_up
#     merge_idxs = persistence < threshold
#     entries[entry_idxs[merge_idxs]] = e_up

#     pers_pairs.extend([[entry_idx, ref_idx] for entry_idx in entry_idxs[merge_idxs]])
#     pers_pairs = np.array(pers_pairs)

#     return entries, pers_pairs
