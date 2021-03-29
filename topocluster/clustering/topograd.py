from __future__ import annotations
import math
import pdb
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from tqdm import tqdm

from topocluster.clustering.topograd_orig import newI1, rips_graph, topoclustergrad
from topocluster.utils.torch_ops import compute_density_map, compute_rips, rbf

from .tomato import cluster, tomato

__all__ = [
    "TopoGradFn",
    "TopoGradLoss",
]


class TopoGradFn(Function):
    @staticmethod
    def forward(
        ctx: Any,
        pc: Tensor,
        k_kde: int,
        k_rips: int,
        scale: float,
        destnum: int,
        **kwargs,
    ) -> Tensor:
        kde_dists, kde_idxs = compute_density_map(pc, k_kde, scale)
        sorted_idxs = torch.argsort(kde_dists, descending=False)
        kde_dists_sorted = kde_dists[sorted_idxs]
        # kde_idxs_sorted = kde_idxs[sorted_idxs]
        kde_idxs_sorted = newI1(kde_idxs, sorted_idxs)
        pc = pc[sorted_idxs]
        rips_idxs = compute_rips(pc, k_rips)
        rips_idxs = torch.tensor(rips_graph(pc.numpy(), k_rips)[1])
        pers_pairs = np.array(cluster(kde_dists.numpy(), rips_idxs.numpy(), 1)[1])

        see = []
        for i in pers_pairs:
            if (i == np.array([-1, -1])).all():
                pass
            else:
                see.append(i)
        see = np.array(see)
        pd_pairs = []
        for i in np.unique(see[:, 0]):
            pd_pairs.append(
                [
                    see[np.where(see[:, 0] == i)[0]][0, 0],
                    max(see[np.where(see[:, 0] == i)[0]][:, 1]),
                ]
            )
        pd_pairs = torch.tensor(pd_pairs)
        oripd = kde_dists_sorted[pd_pairs]
        pers_idxs_sorted = torch.argsort(oripd[:, 0] - oripd[:, 1])

        changing = pers_idxs_sorted[:-destnum]
        nochanging = pers_idxs_sorted[-destnum:-1]

        biggest = oripd[pers_idxs_sorted[-1]]
        dest = torch.tensor([biggest[0], biggest[1]])
        changepairs = pd_pairs[changing]
        nochangepairs = pd_pairs[nochanging]
        pd11 = kde_dists[changepairs]

        weakdist = torch.sum(pd11[:, 0] - pd11[:, 1]) / math.sqrt(2)
        strongdist = torch.sum(torch.norm(kde_dists_sorted[nochangepairs] - dest, dim=1))

        ctx.pc = pc
        ctx.idxs_kde = kde_idxs_sorted
        ctx.dists_kde = kde_dists_sorted
        ctx.scale = scale
        ctx.changepairs = changepairs
        ctx.nochangepairs = nochangepairs
        ctx.dest = dest

        return torch.as_tensor(weakdist + strongdist), rips_idxs

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        pc = ctx.pc
        idxs_kde = ctx.idxs_kde
        dists_kde = ctx.dists_kde
        scale = ctx.scale
        changepairs = ctx.changepairs
        nochangepairs = ctx.nochangepairs
        dest = ctx.dest
        grad_input = torch.zeros_like(pc)

        #  Compute the gradient for changing pairs
        pc_cp_tiled = pc[changepairs][:, :, None]
        coeff_cp_pre = math.sqrt(2) / len(changepairs)  # type: ignore
        coeff_cp = coeff_cp_pre * rbf(
            x=pc_cp_tiled, y=pc[idxs_kde[changepairs]], scale=scale, dim=-1
        )
        direction_cp = pc_cp_tiled - pc[idxs_kde[changepairs]]
        grad_cp = direction_cp * coeff_cp[..., None]
        grad_cp[:, 1] *= -1
        grad_input[idxs_kde[changepairs]] = grad_cp

        #  Compute the gradient for non-changing pairs
        dists = dists_kde[nochangepairs] - dest
        coeff_ncp_pre = (1 / torch.norm(dists) * dists / scale / len(nochangepairs))[..., None]
        pc_ncp_tiled = pc[nochangepairs][:, :, None]
        coeff_ncp = coeff_ncp_pre * rbf(
            x=pc_ncp_tiled, y=pc[idxs_kde[nochangepairs]], scale=scale, dim=-1
        )
        direction_ncp = pc_ncp_tiled - pc[idxs_kde[nochangepairs]]
        grad_ncp = direction_ncp * coeff_ncp[..., None]
        grad_input[idxs_kde[nochangepairs]] = grad_ncp

        return grad_input, None, None, None, None


class TopoGradLoss(nn.Module):
    def __init__(self, k_kde: int, k_rips: int, scale: float, destnum: int) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum

    def forward(self, x: Tensor) -> Tensor:
        # return topoclustergrad.apply(x, self.k_kde, self.k_rips, self.scale, self.destnum)
        return TopoGradFn.apply(x, self.k_kde, self.k_rips, self.scale, self.destnum)


class TopoGradCluster:
    labels: np.ndarray
    pers_pairs: np.ndarray
    split_indces: np.ndarray

    def __init__(
        self,
        destnum: int,
        k_kde: int = 10,
        k_rips: int = 10,
        scale: float = 0.5,
        merge_threshold: float = 1,
        iters: int = 100,
        lr: float = 1.0e-3,
        optimizer_cls=torch.optim.AdamW,
        **optimizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.k_kde = k_kde
        self.k_rips = k_rips
        self.scale = scale
        self.destnum = destnum
        self.merge_threshold = merge_threshold
        self.lr = lr
        self.iters = iters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self._loss_fn = TopoGradLoss(k_kde=k_kde, k_rips=k_rips, scale=scale, destnum=destnum)

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(self.pers_pairs[:, 0], self.pers_pairs[:, 1], s=15, c="orange")  # type: ignore[arg-type]
        ax.plot(np.array([0, 1]), np.array([0, 1]), c="black", alpha=0.6)  # type: ignore[call-arg]
        ax.set_xlabel("Death")
        ax.set_ylabel("Birth")
        ax.set_title("Persistence Diagram")

        return fig

    def fit(self, x: Tensor | np.ndarray) -> TopoGradCluster:
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x).requires_grad_(True)
        else:
            x = x.cpu().detach().clone().requires_grad_(True)
        optimizer = self.optimizer_cls((x,), lr=self.lr)
        with tqdm(desc="topograd", total=self.iters) as pbar:
            for _ in range(self.iters):
                loss = self._loss_fn(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update()

        clusters, pers_pairs = tomato(
            x.detach().numpy(),
            k_kde=self.k_kde,
            k_rips=self.k_rips,
            scale=self.scale,
            threshold=self.merge_threshold,
        )
        cluster_labels = np.empty(x.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k
        self.labels = cluster_labels
        self.pers_pairs = pers_pairs

        return self

    def fit_predict(self, x: Tensor | np.ndarray) -> np.ndarray:
        return self.fit(x).labels

    def predict(self, x: Tensor) -> np.ndarray:
        return self.labels
