from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import warnings

from faiss import IndexFlatL2
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
import umap

from topocluster.utils.numpy_ops import compute_density_map, compute_rips, rbf

from .tomato import cluster

__all__ = [
    "TopoGradFn",
    "TopoGradLoss",
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

    # def forward(
    #     ctx: Any,
    #     pc: Tensor,
    #     dists_kde: np.ndarray[np.float32],
    #     idxs_kde: np.ndarray[np.int64],
    #     pers_pairs: np.ndarray[np.int64],
    #     scale: float,
    #     destnum: int,
    # ) -> Tensor:
    #     pc_np = pc.detach().cpu().numpy()

    #     ctx.pc = pc_np
    #     ctx.destnum = destnum
    #     ctx.idxs_kde = idxs_kde
    #     ctx.dists_kde = dists_kde
    #     ctx.pers_pairs = pers_pairs
    #     ctx.scale = scale

    #     return pc

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
