from __future__ import annotations
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder
from topocluster.utils.torch_ops import compute_density_map, compute_rips

from .tomato import Tomato, cluster, tomato

__all__ = ["topograd_loss", "TopoGrad"]


def topograd_loss(pc: Tensor, k_kde: int, k_rips: int, scale: float, destnum: int) -> Tensor:
    kde_dists, _ = compute_density_map(pc, k_kde, scale)

    sorted_idxs = torch.argsort(kde_dists, descending=False)
    kde_dists_sorted = kde_dists[sorted_idxs]
    pc_sorted = pc[sorted_idxs]

    rips_idxs = compute_rips(pc_sorted, k_rips)
    _, pers_pairs = cluster(
        density_map=kde_dists_sorted.detach().cpu().numpy(),
        rips_idxs=rips_idxs.cpu().numpy(),
        threshold=1.0,
    )

    pers_pairs = torch.as_tensor(pers_pairs, device=pc.device, dtype=torch.long)
    seen = pers_pairs[~torch.all(pers_pairs == -1, dim=1)]

    pd_pairs = []
    for i in torch.unique(seen[:, 0]):
        pd_pairs.append(
            [
                seen[torch.where(seen[:, 0] == i)[0]][0, 0],
                max(seen[torch.where(seen[:, 0] == i)[0]][:, 1]),
            ]
        )
    pd_pairs = torch.as_tensor(pd_pairs, device=pc.device)
    oripd = kde_dists_sorted[pd_pairs]
    pers_idxs_sorted = torch.argsort(oripd[:, 0] - oripd[:, 1])

    changing = pers_idxs_sorted[:-destnum]
    nochanging = pers_idxs_sorted[-destnum:-1]

    biggest = oripd[pers_idxs_sorted[-1]]
    dest = torch.as_tensor([biggest[0], biggest[1]], device=pc.device)
    changepairs = pd_pairs[changing]
    nochangepairs = pd_pairs[nochanging]
    pd11 = kde_dists_sorted[changepairs]

    weakdist = torch.sum(pd11[:, 0] - pd11[:, 1]) / math.sqrt(2)
    strongdist = torch.sum(torch.norm(kde_dists_sorted[nochangepairs] - dest, dim=1))
    return weakdist + strongdist


class TopoGrad(Tomato):
    destnum: int

    def __init__(self, k_kde: int, k_rips: int, scale: float, threshold: float):
        super().__init__(k_kde=k_kde, k_rips=k_rips, scale=scale, threshold=threshold)

    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.destnum = datamodule.num_subgroups * datamodule.num_classes

    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        if not hasattr(self, "destnum"):
            raise AttributeError(
                "destnum has not yet been set. Please call 'build' before calling 'get_loss'"
            )
        loss = topograd_loss(
            pc=x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, destnum=self.destnum
        )
        return {"saliency_loss": loss}


class TopoGradCluster:
    """
    Variant of topograd designed to be applied to static embeddings rather than used to train an
    embedding network.
    """

    labels: np.ndarray
    pers_pairs: np.ndarray

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
                loss = topograd_loss(
                    pc=x,
                    k_kde=self.k_kde,
                    k_rips=self.k_rips,
                    scale=self.scale,
                    destnum=self.destnum,
                )
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
