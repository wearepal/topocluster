from __future__ import annotations
import math
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from topocluster.clustering.utils import l2_centroidal_distance
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

    pers_pairs = torch.as_tensor(pers_pairs, device=pc.device)
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

    def __init__(
        self,
        k_kde: int,
        k_rips: int,
        scale: float,
        threshold: float,
        iters: int = 0,
        lr: float = 1e-3,
    ):
        super().__init__(k_kde=k_kde, k_rips=k_rips, scale=scale, threshold=threshold)
        self.iters = iters
        self.optimizer_cls = torch.optim.AdamW
        self.lr = lr

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

    def __call__(self, x: Tensor, threshold: float | None = None) -> tuple[Tensor, Tensor]:
        threshold = self.threshold if threshold is None else threshold
        # Run topograd on the embedding (without backpropagating through the network)
        if self.iters > 0:
            # Avoid modifying the original embedding
            x = x.detach().clone()
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
            x.detach().cpu().numpy(),
            k_kde=self.k_kde,
            k_rips=self.k_rips,
            scale=self.scale,
            threshold=self.threshold,
        )
        cluster_labels = np.empty(x.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k
        self.labels = cluster_labels
        self.pers_pairs = torch.as_tensor(pers_pairs)

        cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.long)
        centroids = x[list(clusters.keys())]
        soft_labels = l2_centroidal_distance(x=x, centroids=centroids)
        hard_labels = cluster_labels

        return hard_labels, soft_labels
