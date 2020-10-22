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

    # find entries of U intersecting S whose roots are less than τ -prominent; Merge those into e_i
    e_up = find_entry_idx_by_point(entries, upper_star_idxs[0])
    for j in range(1, len(upper_star_idxs)):
        star_idx = find_entry_idx_by_point(entries, upper_star_idxs[j])
        # Let e_j be the entry of U containing j;
        if density_map[star_idx] > density_map[e_up]:
            e_up = star_idx
    # find entry ¯e of U intersecting S whose root is highest
    for j in upper_star_idxs:
        entry_idx = find_entry_idx_by_point(entries, j)
        # merge e_i into ¯e if the prominence of the root of e_i is less than τ}
        if (e_up != entry_idx) and ((density_map[entry_idx] - density_map[ref_idx]) < threshold):
            ggg.append([entry_idx, ref_idx])
            # Remove entry ej from U and attach it to ei;
            entries[e_up].extend(entries[entry_idx])
            entries.pop(entry_idx)

    return entries, ggg


@torch.jit.script
def topocluster(
    density_map: Tensor, rips_indexes: Tensor, threshold: float
) -> Tuple[Tensor, Tensor]:
    # Sort the index set so that f1 < f2 < ... <f3
    sorted_idxs = density_map.argsort()
    density_map = density_map[sorted_idxs]
    rips_indexes = rips_indexes[sorted_idxs]

    N = density_map.size(0)
    # initialise the union-find data-structure
    entries: Dict[int, List[int]] = {N - 1: [N - 1]}
    lifespans = [[-1, -1]]
    # iterate in reverse order, from largest f_i to smallest f_i
    inds = range(N - 2, -1, -1)

    for i in inds:
        nbr_idxs = rips_indexes[i]
        # compute the upper star Si = {(i, j1), · · · , (i, jk)} of vertex i in R_δ(L);
        us_idxs = nbr_idxs[nbr_idxs >= i]
        # check whether vertex i is a local maximum of f within R_δ
        if len(us_idxs) == 1:
            entries[i] = [i]
        # vertex i is not a local maximum of f within R_δ
        else:
            # approximate the gradient of the underlying probability density function by connecting
            # i to its neighbour in the graph with the highest function value
            grad_i = torch.max(us_idxs)
            # Attach vertex i to the tree t containing g(i)
            entry_index = find_entry_idx_by_point(entries=entries, point_idx=grad_i)
            entries[entry_index].append(i)
            # Check for merges and update the union-find data-structure
            entries, lifespans_i = merge(
                density_map=density_map,
                entries=entries,
                ref_idx=i,
                upper_star_idxs=us_idxs,
                threshold=threshold,
            )
            if len(lifespans_i) > 1:
                lifespans.extend(lifespans_i)

    entries_t = torch.empty(N, dtype=torch.long)
    for i, nbrs in enumerate(entries.values()):
        entries_t[nbrs] = i

    return entries_t, torch.tensor(lifespans)
