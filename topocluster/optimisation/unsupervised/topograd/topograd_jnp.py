from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from fast_soft_sort.jax_ops import soft_rank
from jax import jit, vmap

# __all__ = ["TopoCluster", "compute_vrc", "compute_barcode", "topograd", "compute_density_map"]


def topograd(
    pc: jnp.array[jnp.float32],
    clusters: jnp.array[jnp.float32],
    dists_kde: jnp.array[jnp.float32],
    inds_kde: jnp.array[jnp.int32],
    destnum: int,
    scale: float,
) -> jnp.array:
    grads = jnp.zeros_like(pc)

    see = jnp.array([elem for elem in clusters if (elem != jnp.array([-1, -1])).any()])
    result = []

    for i in np.unique(see[:, 0]):
        result.append([see[see[:, 0] == i][0, 0], max(see[see[:, 0] == i][:, 1])])
    result = jnp.array(result)
    pdpairs = result
    oripd = dists_kde[result]
    sorted_idxs = jnp.argsort(oripd[:, 0] - oripd[:, 1])
    changing = sorted_idxs[:-destnum]
    nochanging = sorted_idxs[-destnum:]
    biggest = oripd[sorted_idxs[-1]]
    dest = jnp.array([biggest[0], biggest[1]])
    changepairs = pdpairs[changing]
    nochangepairs = pdpairs[nochanging]

    #  Compute the gradient for changing pairs
    pc_cp_tiled = pc[changepairs][:, :, None]
    coeff_cp_pre = jnp.sqrt(2) / len(changepairs)
    coeff_cp = coeff_cp_pre * rbf(x=pc_cp_tiled, y=pc[inds_kde[changepairs]], scale=scale, axis=-1)
    direction_cp = -pc[inds_kde[changepairs]]
    grad_cp = direction_cp * coeff_cp[..., None]
    grad_cp[:, 1] *= -1
    grads[inds_kde[changepairs]] = grad_cp

    #  Compute the gradient for non-changing pairs
    dists = dists_kde[nochangepairs] - dest
    coeff_ncp_pre = (1 / jnp.linalg.norm(dists) * dists / scale / len(nochangepairs))[..., None]
    pc_ncp_tiled = pc[nochangepairs][:, :, None]
    coeff_ncp = coeff_ncp_pre * rbf(
        x=pc_ncp_tiled, y=pc[inds_kde[nochangepairs]], scale=scale, axis=-1
    )
    direction_ncp = pc_ncp_tiled - pc[inds_kde[nochangepairs]]
    grad_ncp = direction_ncp * coeff_ncp[..., None]
    grads[inds_kde[nochangepairs]] = grad_ncp

    return grads


def knn(pc: jnp.array, k: int, soft: bool = False) -> Tuple[jnp.array, jnp.array]:
    D_ij = jnp.square(pc[None] - pc[:, None]).sum(-1)
    if soft:
        indKNN = soft_rank(D_ij)[:, :k]
    else:
        indKNN = jnp.argsort(D_ij, axis=-1)[:, :k]
    return D_ij, indKNN


def rbf(x: jnp.array, y: jnp.array, scale: float, axis: int = -1) -> jnp.array:
    return jnp.exp(-jnp.linalg.norm(x - y, axis=axis) ** 2 / scale)


def compute_rips(pc: jnp.array, k: int) -> Tuple[jnp.array, jnp.array]:
    return knn(pc=pc, k=k)


def compute_density_map(pc: jnp.array, k: int, scale: float) -> Tuple[jnp.array, jnp.array]:
    """Compute the k-nearest neighbours kernel density estimate."""
    dists, inds = knn(pc=pc, k=k)
    result = np.sum(np.exp(-dists / scale), axis=1) / (k * scale)
    return result / max(result), inds  # / max(result)


@jit
def find_entry_idx_by_point(entries: Dict[int, List[int]], point_idx: int) -> int:
    for index, entry in entries.items():
        for i in entry:
            if i == point_idx:
                return int(index)
    return 0


# @jit
# def topocluster(
#     density_map: jnp.array, rips_indexes: jnp.array, threshold: float
# ) -> Tuple[Dict[int, jnp.array], jnp.array]:
#     clusters = jnp.array([[-1, -1]])
#     #  initialize the entries with the final index pointing only to itself
#     entries = {density_map.shape[0] - 1: jnp.array([density_map.shape[0] - 1])}

#     idxs = jnp.arange(rips_indexes.shape[0])
#     us_idxs_idxs = rips_indexes >= idxs[:, None]
#     is_maximum = us_idxs_idxs.sum(1) == 1
#     # entries[is_maximum] = inds[is_maximum]
#     lopts = (us_idxs_idxs * rips_indexes).max(axis=1)

#     for i in np.arange(density_map.shape[0] - 2, -1, -1):
#         if is_maximum[i]:
#             entries[i] = jnp.array([i])
#         else:
#             entry_idx = find_entry_idx_by_point(entries, lopts[i])  #  find the index of the
#             entries[entry_idx] = jnp.append(entries[entry_idx], i)
#             entries, kkk = merge(
#                 density_map, entries, i, rips_indexes[i][us_idxs_idxs[i]], threshold
#             )
#             if len(kkk) > 1:
#                 clusters = jnp.append(clusters, kkk, axis=0)

#     return entries, clusters


@jit
def topocluster(
    density_map: jnp.array, rips_indexes: jnp.array, threshold: float
) -> Tuple[Dict[int, jnp.array], jnp.array]:
    clusters = jnp.array([[-1, -1]])
    #  initialize the entries with the final index pointing only to itself
    entries = {density_map.shape[0] - 1: jnp.array([density_map.shape[0] - 1])}
    for i in jnp.arange(density_map.shape[0] - 2, -1, -1):
        nbr_idxs = rips_indexes[i]
        upper_star_idxs = nbr_idxs[nbr_idxs >= i]
        if upper_star_idxs.sum() == 1:  # i is a local maximum
            entries[i] = jnp.array([i])  #  create an entry for the local maximum
        else:
            g_i = jnp.max(upper_star_idxs)  #  find the maximum index
            entry_idx = find_entry_idx_by_point(entries, g_i)
            entries[entry_idx] = jnp.append(entries[entry_idx], i)
            entries, kkk = merge(density_map, entries, i, upper_star_idxs, threshold)
            if len(kkk) > 1:
                clusters = jnp.append(clusters, kkk, axis=0)

    return entries, cluster


@jit
def merge(
    density_map: jnp.array,
    entries: Dict[int, List[int]],
    ref_idx: int,
    upper_star_idxs: List[int],
    threshold: float,
) -> Tuple[Dict[int, List[int]], jnp.array[np.int]]:
    ggg = jnp.array([[-1, -1]])

    for j in range(len(upper_star_idxs)):
        star_idx = find_entry_idx_by_point(entries, upper_star_idxs[j])
        if (j == 0) or (density_map[star_idx] > density_map[e_up]):
            e_up = star_idx

    for j in upper_star_idxs:
        entry_idx = int(find_entry_idx_by_point(entries, j))

        if (e_up != entry_idx) and (density_map[entry_idx] - density_map[ref_idx] < threshold):
            ggg = jnp.append(ggg, jnp.array([[entry_idx, ref_idx]]), axis=0)
            entries[e_up] = jnp.append(entries[e_up], entries[entry_idx])
            entries.pop(entry_idx)

    return entries, ggg
