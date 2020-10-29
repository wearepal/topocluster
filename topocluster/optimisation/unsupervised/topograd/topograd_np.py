from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap
from faiss import IndexFlatL2
from numba import jit
from torch import Tensor
from torch.autograd import Function

__all__ = [
    "Tomato",
    "TopoGradFn",
    "TopoGradLoss",
    "compute_density_map",
    "compute_rips",
    "tomato",
    "topograd",
]


class TopoGradFn(Function):
    @staticmethod
    def forward(
        ctx: Any, pc: Tensor, k_kde: int, k_rips: int, scale: float, destnum: int, **kwargs
    ) -> Tensor:
        pc_np = pc.detach().cpu().numpy()
        dists_kde, idxs_kde = compute_density_map(pc_np, k_kde, scale)
        dists_kde = dists_kde.astype(float)
        sorted_idxs = np.argsort(dists_kde)
        idxs_kde_sorted = idxs_kde[sorted_idxs]
        dists_kde_sorted = dists_kde[sorted_idxs]
        pc_np = pc_np[sorted_idxs]
        _, rips_idxs = compute_rips(pc_np, k_rips)
        _, pers_pairs = cluster(dists_kde, rips_idxs, 1)

        ctx.pc = pc_np
        ctx.destnum = destnum
        ctx.idxs_kde = idxs_kde_sorted
        ctx.dists_kde = dists_kde_sorted
        ctx.pers_pairs = pers_pairs
        ctx.scale = scale

        return pc

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_input = np.zeros_like(grad_output)
        pc = ctx.pc
        pers_pairs = ctx.pers_pairs
        destnum = ctx.destnum
        idxs_kde = ctx.idxs_kde
        dists_kde = ctx.dists_kde
        scale = ctx.scale

        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])
        result = []

        for i in np.unique(see[:, 0]):
            result.append([see[see[:, 0] == i][0, 0], max(see[see[:, 0] == i][:, 1])])
        result = np.array(result)
        pdpairs = result
        oripd = dists_kde[result]
        sorted_idxs = np.argsort(oripd[:, 0] - oripd[:, 1])
        changing = sorted_idxs[:-destnum]
        nochanging = sorted_idxs[-destnum:]
        biggest = oripd[sorted_idxs[-1]]
        dest = np.array([biggest[0], biggest[1]])
        changepairs = pdpairs[changing]
        nochangepairs = pdpairs[nochanging]

        #  Compute the gradient for changing pairs
        pc_cp_tiled = pc[changepairs][:, :, None]
        coeff_cp_pre = np.sqrt(2) / len(changepairs)
        coeff_cp = coeff_cp_pre * rbf(
            x=pc_cp_tiled, y=pc[idxs_kde[changepairs]], scale=scale, axis=-1
        )
        direction_cp = pc_cp_tiled - pc[idxs_kde[changepairs]]
        grad_cp = direction_cp * coeff_cp[..., None]
        grad_cp[:, 1] *= -1
        grad_input[idxs_kde[changepairs]] = grad_cp

        #  Compute the gradient for non-changing pairs
        dists = dists_kde[nochangepairs] - dest
        coeff_ncp_pre = (1 / np.linalg.norm(dists) * dists / scale / len(nochangepairs))[..., None]
        pc_ncp_tiled = pc[nochangepairs][:, :, None]
        coeff_ncp = coeff_ncp_pre * rbf(
            x=pc_ncp_tiled, y=pc[idxs_kde[nochangepairs]], scale=scale, axis=-1
        )
        direction_ncp = pc_ncp_tiled - pc[idxs_kde[nochangepairs]]
        grad_ncp = direction_ncp * coeff_ncp[..., None]
        grad_input[idxs_kde[nochangepairs]] = grad_ncp

        grad_input = torch.as_tensor(grad_input)

        return grad_input, None, None, None, None


class TopoGradLoss(nn.Module):
    def __init__(self, k_kde: int, k_rips: int, scale: float, destnum: int) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum

    def forward(self, x: Tensor) -> Tensor:
        # return x
        return TopoGradFn.apply(x, self.k_kde, self.k_rips, self.scale, self.destnum)


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

    def fit(
        self, x: Union[np.ndarray[np.float32], Tensor], threshold: float = 1
    ) -> Tuple[Tensor, Tensor]:
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
        ) -> Tuple[Dict[int, np.ndarray[np.int64]], np.ndarray[np.float32]]:
            return tomato(
                _x,
                self.k_kde,
                self.k_rips,
                self.scale,
                threshold,
            )

        clusters: Mapping[int, Union[List[np.int64], np.array[np.int64]]]
        pers_pairs: np.ndarray[np.float32]

        if batch_size < num_samples:
            clusters = defaultdict(list)
            pers_pairs_ls: List[np.array[np.float32]] = []
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
        self, x: Union[np.ndarray[np.float32], Tensor], threshold: float = 1
    ) -> Tensor:
        self.fit(x=x, threshold=threshold)
        return self._labels


def rbf(x: np.ndarray, y: np.ndarray, scale: float, axis: int = -1) -> np.ndarray[np.float32]:
    "Compute the distance between two vectors using an RBF kernel."
    return np.exp(-np.linalg.norm(x - y, axis=axis) ** 2 / scale)


def compute_rips(pc: np.ndarray[np.float32], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """"Compute the delta-Rips Graph."""
    pc = pc.astype(np.float32)
    cpuindex = IndexFlatL2(pc.shape[1])
    cpuindex.add(pc)

    return cpuindex.search(pc, k)


def tomato(
    pc: np.ndarray, k_kde: int, k_rips: int, scale: float, threshold: float
) -> Tuple[Dict[int, np.ndarray[np.int32]], np.ndarray[np.float32]]:
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


def topograd(
    pc: np.ndarray[np.float32],
    k_kde: int,
    k_rips: int,
    scale: float,
    destnum: int,
    lr: float,
    iters: int,
) -> Tuple[np.ndarray[np.float32], Optional[np.ndarray[np.float32]]]:
    dists_kde = None
    result = None
    for _ in range(iters):
        dists_kde, idxs_kde = compute_density_map(pc, k_kde, scale)
        dists_kde = dists_kde.astype(float)
        sorted_idxs = np.argsort(dists_kde)
        idxs_kde = idxs_kde[sorted_idxs]
        dists_kde = dists_kde[sorted_idxs]
        pc = pc[sorted_idxs]
        _, vrc_idxs = compute_rips(pc, k_rips)
        _, pers_pairs = cluster(dists_kde, vrc_idxs, 1)
        see = np.array([elem for elem in pers_pairs if (elem != np.array([-1, -1])).any()])

        result = []
        for i in np.unique(see[:, 0]):
            result.append(
                [
                    see[see[:, 0] == i][0, 0],
                    max(see[see[:, 0] == i][:, 1]),
                ]
            )
        result = np.array(result)
        pdpairs = result
        oripd = dists_kde[result]
        sorted_idxs = np.argsort(
            oripd[:, 0] - oripd[:, 1]
        )  #  sort the pairs in order of increasin persistence
        changing = sorted_idxs[:-destnum]
        nochanging = sorted_idxs[-destnum:]
        biggest = oripd[sorted_idxs[-1]]
        dest = np.array([biggest[0], biggest[1]])
        changepairs = pdpairs[changing]
        nochangepairs = pdpairs[nochanging]
        #  Gradient descent
        N = len(changepairs)
        for i in changepairs:
            coeff0 = np.sqrt(2) / N * rbf(x=pc[i[0]], y=pc[idxs_kde[i[0]]], scale=scale)
            direction0 = pc[i[0]] - pc[idxs_kde[i[0]]]
            pc[idxs_kde[i[0]]] -= lr * multiply(direction0, scalar=coeff0)

            coeff1 = -np.sqrt(2) / N * rbf(x=pc[i[1]], y=pc[idxs_kde[i[1]]], scale=scale)
            direction1 = pc[i[1]] - pc[idxs_kde[i[1]]]
            pc[idxs_kde[i[1]]] -= lr * multiply(direction1, scalar=coeff1)

        pd11 = dists_kde[changepairs]
        print("weakening dist: " + str(np.sum(pd11[:, 0] - pd11[:, 1]) / 2))
        sal = 0

        N = len(nochangepairs)
        for i in nochangepairs:
            dist = np.linalg.norm(dists_kde[i] - dest)
            if dist != 0:
                for j in range(0, 2):
                    coeff = (
                        1
                        / dist
                        * (dists_kde[i[j]] - dest[j])
                        / scale
                        / N
                        * rbf(x=pc[i[j]], y=pc[idxs_kde[i[j]]], scale=scale)
                    )
                    direction = pc[i[j]] - pc[idxs_kde[i[j]]]
                    pc[idxs_kde[i[j]]] -= lr * multiply(direction, scalar=coeff)

                sal = sal + dist
        print("salient dist: " + str(sal))
    if (dists_kde is not None) and (result is not None):
        dists_kde = dists_kde[result]
    return pc, dists_kde


def multiply(arr: np.ndarray, scalar: np.ndarray[np.float32]) -> np.ndarray:
    """"Multiply an array by a scalar value along the last dimension."""
    for i in range(arr.shape[1]):
        arr[:, i] = arr[:, i] * scalar
    return arr


def compute_density_map(x: np.ndarray, k: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k-nearest neighbours kernel density estimate."""
    x = x.astype(np.float32)
    index = IndexFlatL2(x.shape[1])
    index.add(x)
    values, indexes = index.search(x, k)
    result = np.sum(np.exp(-values / scale), axis=1) / (k * scale)
    return result / max(result), indexes


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
) -> Tuple[Dict[int, List[int]], np.ndarray[np.int64]]:
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
