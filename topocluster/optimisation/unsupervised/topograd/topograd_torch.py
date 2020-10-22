from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.tensor import Tensor


def pairwise_L2sqr(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return (tensor_a - tensor_b) ** 2


def rbf(x: Tensor, y: Tensor, scale: float) -> Tensor:
    return torch.exp(-torch.norm(x - y, axis=1) ** 2 / scale)


def compute_density_map(pc: Tensor, k: int, scale: float) -> Tuple[Tensor, Tensor]:
    dists, inds = knn(pc, k=k, kernel=pairwise_L2sqr)
    dists = torch.sum(torch.exp(-dists / scale), dim=1) / (k * scale)
    return dists / dists.max(), inds


def compute_rips(pc: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    return knn(pc=pc, k=k, kernel=pairwise_L2sqr)


def knn(
    pc: Tensor, k: int, kernel: Callable[[Tensor, Tensor], Tensor] = pairwise_L2sqr
) -> Tuple[Tensor, Tensor]:
    X_i = pc[:, None, :]
    X_j = pc[None, :, :]  # (1, N, 2)
    D_ij = kernel(X_i, X_j).sum(-1)  # (M**2, N) symbolic matrix of distances
    indKNN = D_ij.topk(k=k, dim=1, largest=False)[1]
    # indKNN = soft_rank(D_ij, direction="ASCENDING", regularization_strength=0.01)
    return D_ij, indKNN


@torch.jit.script
def find_entry_idx_by_point(entries: Dict[int, List[int]], point_idx: int) -> int:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return index
    return point_idx


def merge(
    density_map: Tensor,
    entries: Dict[int, List[int]],
    ref_idx: int,
    upper_star_idxs: Tensor,
    threshold: float,
) -> Tuple[Dict[int, List[int]], List[List[int]]]:
    ggg = [[-1, -1]]

    e_up = find_entry_idx_by_point(entries, upper_star_idxs[0])
    for j in range(1, len(upper_star_idxs)):
        star_idx = find_entry_idx_by_point(entries, upper_star_idxs[j])
        if density_map[star_idx] > density_map[e_up]:
            e_up = star_idx

    for j in upper_star_idxs:
        entry_idx = find_entry_idx_by_point(entries, j)

        if (e_up != entry_idx) and (density_map[entry_idx] - density_map[ref_idx] < threshold):
            ggg.append([entry_idx, ref_idx])
            entries[e_up].extend(entries[entry_idx])
            entries.pop(entry_idx)

    return entries, ggg


@torch.jit.script
def topocluster(
    density_map: Tensor, rips_indexes: Tensor, threshold: float
) -> Tuple[Dict[int, List[int]], Tensor]:
    N = density_map.size(0)
    # entries = defaultdict(list, {N - 1: [N - 1]})
    entries: Dict[int, List[int]] = {N - 1: [N - 1]}
    clusters = [[-1, -1]]
    # inds = torch.arange(N - 2, -1, -1)
    inds = range(N - 2, -1, -1)

    # us_idxs_idxs = rips_indexes[inds] >= inds[:, None]
    # is_maximum = us_idxs_idxs.sum(dim=1) == 1
    # us_idxs = us_idxs_idxs * rips_indexes[inds]
    # local_opts = us_idxs.max(dim=1)[0]

    for i in inds:
        nbr_idxs = rips_indexes[i]
        us_idxs = nbr_idxs[nbr_idxs >= i]
        is_maximum = len(us_idxs) == 1
        if is_maximum:
            entries[i] = [i]
        else:
            g_i = torch.max(us_idxs)
            entry_index = find_entry_idx_by_point(entries=entries, point_idx=g_i)
            entries[entry_index].append(i)
            entries, kkk = merge(
                density_map=density_map,
                entries=entries,
                ref_idx=i,
                upper_star_idxs=us_idxs,
                threshold=threshold,
            )
            if len(kkk) > 1:
                clusters.extend(kkk)

    return entries, torch.tensor(clusters)

