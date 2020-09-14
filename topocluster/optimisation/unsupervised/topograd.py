from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
from faiss import IndexFlatL2
from numba import jit
import torch

import umap
from torch import Tensor


__all__ = ["TopoCluster"]


class TopoCluster:
    def __init__(
        self,
        k_kde: int = 100,
        k_vrc: int = 15,
        scale: float = 0.5,
        batch_size: Optional[int] = None,
        umap_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.k_kde = k_kde
        self.k_vrc = k_vrc
        self.scale = scale
        self.batch_size = batch_size
        self.reducer = umap.UMAP(**umap_kwargs) if umap_kwargs is not None else None

    @staticmethod
    def plot_pd(barcode: Union[Tensor, np.ndarray[np.float]], dpi: int = 100) -> plt.Figure:
        fig, ax = plt.subplots(dpi=dpi)
        ax.scatter(barcode[:, 1], barcode[:, 0], s=15, c="orange")  # type: ignore[arg-type]
        ax.plot(np.array([0, 1]), np.array([0, 1]), c="black", alpha=0.6)  # type: ignore[call-arg]
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("Persistence Diagram")

        return fig

    def fit(
        self, x: Union[np.ndarray[np.float], Tensor], threshold: float = 1
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
        ) -> Tuple[Dict[int, np.ndarray[np.int]], np.ndarray[np.float]]:
            return compute_barcode(
                point_cloud=_x,
                k_kde=self.k_kde,
                k_vrc=self.k_vrc,
                scale=self.scale,
                threshold=threshold,
            )

        if batch_size < num_samples:
            clusters, barcode = defaultdict(list), []
            batches = np.array_split(x, indices_or_sections=batch_size, axis=0)
            for batch in batches:
                clusters_b, pd_b = _partialled(batch)
                for key, values in clusters_b.items():
                    clusters[key].extend(values)
                barcode.append(pd_b)

            barcode = np.concatenate(barcode, axis=0)

        else:
            clusters, barcode = _partialled(x)

        cluster_labels = np.empty(x.shape[0])
        for k, y in enumerate(clusters.values()):
            cluster_labels[y] = k

        cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.int32)
        barcode = torch.as_tensor(barcode, dtype=torch.float32)

        return cluster_labels, barcode


def compute_barcode(
    point_cloud: np.ndarray, k_kde: int, k_vrc: int, scale: float, threshold: float
) -> Tuple[Dict[int, np.ndarray[np.int]], np.ndarray[np.float]]:
    point_cloud = point_cloud.astype(float)
    #  Compute the k-NN KDE
    density_map, _ = compute_density_map(point_cloud, k_kde, scale)
    density_map = density_map.astype(float)

    sorted_idxs = np.argsort(density_map)
    density_map_sorted = density_map[sorted_idxs]
    point_cloud = point_cloud[sorted_idxs]

    _, vrc_indexes = compute_vrc(point_cloud, k=k_vrc)
    ddd, ddqq = cluster(density_map_sorted, vrc_indexes, threshold=threshold)
    if threshold == 1:
        see = np.array([elem for elem in ddqq if (elem != np.array([-1, -1])).any()])
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
            for key, value in ddd.items():
                ddd[key] = sorted_idxs[value]
        else:
            warnings.warn("Clustering unsuccessful; consider expanding the VRC neighbourhood.")
        return ddd, density_map_sorted[result]
    else:
        for key, value in ddd.items():
            ddd[key] = sorted_idxs[value]
        return ddd, np.array([[0, 0]])


def major(
    pc: np.ndarray[np.float32], k_vrc: int, k_kde: int, scale: float, destnum: int, lr: float, epochs: int
):
    for _ in range(epochs):
        f, I1 = compute_density_map(pc, k_kde, scale)
        f = f.astype(float)
        pc = pc.astype(float)

        sorted_idxs = np.argsort(f)
        I1 = I1[sorted_idxs]
        f = f[sorted_idxs]
        pc_sorted = pc[sorted_idxs]
        lims, vrc_indexes = compute_vrc(pc_sorted, k_vrc)
        _, ddqq = cluster(f, vrc_indexes, lims, 1)
        see = np.array([elem for elem in ddqq if (elem != np.array([-1, -1])).any()])

        result = []
        for i in np.unique(see[:, 0]):
            result.append(
                [
                    see[np.where(see[:, 0] == i)[0]][0, 0],
                    max(see[np.where(see[:, 0] == i)[0]][:, 1]),
                ]
            )
        result = np.array(result)
        pdpairs = result
        oripd = f[result]
        sorted_idxs = np.argsort(oripd[:, 0] - oripd[:, 1])
        changing = sorted_idxs[:-destnum]
        nochanging = sorted_idxs[-destnum:-1]
        biggest = oripd[sorted_idxs[-1]]
        dest = np.array([biggest[0], biggest[1]])
        changepairs = pdpairs[changing]
        nochangepairs = pdpairs[nochanging]
        #             print(oripd)
        for i in changepairs:
            coeff = (
                np.sqrt(2)
                / len(changepairs)
                * np.exp(-np.linalg.norm(pc[i[0]] - pc[I1[i[0]]], axis=1) ** 2 / scale)
            )
            direction = pc[i[0]] - pc[I1[i[0]]]
            pc[I1[i[0]]] = pc[I1[i[0]]] - lr * multiply(direction, coeff)
            coeff1 = (
                -np.sqrt(2)
                / len(changepairs)
                * np.exp(-np.linalg.norm(pc[i[1]] - pc[I1[i[1]]], axis=1) ** 2 / scale)
            )
            direction1 = pc[i[1]] - pc[I1[i[1]]]
            pc[I1[i[1]]] = pc[I1[i[1]]] - lr * multiply(direction1, coeff1)

        pd11 = f[changepairs]
        print("weakening dist: " + str(np.sum(pd11[:, 0] - pd11[:, 1]) / 2))
        sal = 0
        for i in nochangepairs:
            dist = np.linalg.norm(f[i] - dest)
            if dist == 0:
                pass
            else:
                coeff = (
                    1
                    / dist
                    * (f[i[0]] - dest[0])
                    / scale
                    / len(nochangepairs)
                    * np.exp(-np.linalg.norm(pc[i[0]] - pc[I1[i[0]]], axis=1) ** 2 / scale)
                )
                direction = pc[i[0]] - pc[I1[i[0]]]
                pc[I1[i[0]]] = pc[I1[i[0]]] - lr * multiply(direction, coeff)
                coeff1 = (
                    1
                    / dist
                    * (f[i[1]] - dest[1])
                    / scale
                    / len(nochangepairs)
                    * np.exp(-np.linalg.norm(pc[i[1]] - pc[I1[i[1]]], axis=1) ** 2 / scale)
                )
                direction1 = pc[i[1]] - pc[I1[i[1]]]
                pc[I1[i[1]]] = pc[I1[i[1]]] - lr * multiply(direction1, coeff1)

                sal = sal + dist
        print("salient dist: " + str(sal))

    return pc, f[result]


def multiply(arr: np.ndarray, scalar: float) -> np.ndarray:
    """"Multiply an array by a scalar value along the last dimension."""
    for i in range(arr.shape[1]):
        arr[:, i] = arr[:, i] * scalar
    return arr


def compute_density_map(x: np.ndarray, k: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k-nearest neighbours kernel density estimate."""
    x = x.astype(np.float32)
    index = IndexFlatL2(len(x[0]))
    index.add(x)
    values, indexes = index.search(x, k)
    result = np.sum(np.exp(-values / scale), axis=1) / (k * scale)
    return result / max(result), indexes  # / max(result)


@jit(nopython=True)
def find_entry_idx_by_point(entries: Dict[int, List[int]], point_idx: int) -> Optional[np.int64]:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return np.int64(index)
    return None

@jit(nopython=True)
def cluster(
    f: np.ndarray, knn_indexes: np.ndarray, threshold: float
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    clusters = np.array([[-1, -1]])
    entries = {f.shape[0] - 1: np.array([f.shape[0] - 1])}
    for i in np.arange(f.shape[0] - 2, -1, -1):
        nbr_idxs = knn_indexes[i]
        upper_star_idxs = nbr_idxs[nbr_idxs >= i]
        if upper_star_idxs.size == 1:
            # i is a local maximum
            entries[i] = np.array([i])
        else:
            g_i = np.max(upper_star_idxs)
            entry_idx = find_entry_idx_by_point(entries, g_i)
            entries[entry_idx] = np.append(entries[entry_idx], i)
            entries, kkk = merge(f, entries, i, upper_star_idxs, threshold)
            if len(kkk) > 1:
                clusters = np.append(clusters, kkk, axis=0)
    return entries, clusters


@jit(nopython=True)
def merge(f, entries, i, upper_star_idxs, threshold):
    ggg = np.array([[-1, -1]])

    for j in range(len(upper_star_idxs)):
        star_idx = find_entry_idx_by_point(entries, upper_star_idxs[j])
        if j == 0:
            e_up = star_idx
        elif f[np.int64(star_idx)] > f[np.int64(e_up)]:
            e_up = star_idx

    for j in upper_star_idxs:
        entry_idx = find_entry_idx_by_point(entries, j)

        if (e_up != entry_idx) and (f[np.int64(entry_idx)] - f[np.int64(i)] < threshold):
            ggg = np.append(ggg, np.array([[int(entry_idx), int(i)]]), axis=0)
            entries[e_up] = np.append(entries[e_up], entries[entry_idx])
            entries.pop(entry_idx)

    return entries, ggg


def compute_vrc(point_cloud: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """"Compute the Vietroris-Rips Complex."""
    point_cloud = point_cloud.astype("float32")
    _, dim = point_cloud.shape
    cpuindex = IndexFlatL2(dim)
    cpuindex.add(point_cloud)

    return cpuindex.search(point_cloud, k)
