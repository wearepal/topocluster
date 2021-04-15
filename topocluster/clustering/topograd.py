from __future__ import annotations
import logging
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm

from kit import implements
from topocluster.clustering.common import Clusterer
from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder
from topocluster.utils.torch_ops import compute_density_map, compute_rips

from .tomato import Tomato, cluster, tomato

__all__ = ["topograd_loss", "TopoGrad"]

LOGGER = logging.getLogger(__name__)


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
    if not pd_pairs:
        LOGGER.info(
            "Filtering failed to yield sufficient persistence pairs for computation of "
            "the topological loss. Returning 0 instead."
        )
        return pc.new_zeros(())
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


class TopoGrad(nn.Module, Tomato):
    destnum: int

    def __init__(
        self,
        k_kde: int,
        k_rips: int,
        scale: float,
        threshold: float,
        n_iter: int = 0,
        lr: float = 1e-3,
        add_bias: bool = False
    ):
        super().__init__(k_kde=k_kde, k_rips=k_rips, scale=scale, threshold=threshold)
        self.n_iter = n_iter
        self.optimizer_cls = torch.optim.AdamW
        self.lr = lr
        self.add_bias = add_bias
        self.register_parameter("bias", None)

    @implements(Clusterer)
    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.destnum = datamodule.num_subgroups * datamodule.num_classes
        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(encoder.latent_dim), requires_grad=True)

    @implements(Clusterer)
    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        if not hasattr(self, "destnum"):
            raise AttributeError(
                "destnum has not yet been set. Please call 'build' before calling 'get_loss'"
            )
        if self.bias is not None:
            x = x + self.bias
        loss = topograd_loss(
            pc=x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, destnum=self.destnum
        )
        return {"saliency_loss": loss}

    @implements(nn.Module)
    def forward(self, x: Tensor, threshold: float | None = None) -> Tensor:
        threshold = self.threshold if threshold is None else threshold
        # Run topograd on the embedding (without backpropagating through the network)
        if self.n_iter > 0:
            # Avoid modifying the original embedding
            x = x.detach()
            if self.add_bias is None:
                x = x.clone().requires_grad_(True)
                optimizer = self.optimizer_cls((x,), lr=self.lr)
            else:
                optimizer = self.optimizer_cls((self.bias,), lr=self.lr)
            with tqdm(desc="topograd", total=self.n_iter) as pbar:
                for _ in range(self.n_iter):
                    loss = self.get_loss(x=x)["saliency_loss"]
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
            threshold=threshold,
        )
        cluster_labels = np.empty(x.shape[0])
        for k, v in enumerate(clusters.values()):
            cluster_labels[v] = k
        self.labels = cluster_labels
        self.pers_pairs = torch.as_tensor(pers_pairs)

        cluster_labels = torch.as_tensor(cluster_labels, dtype=torch.long)

        return cluster_labels
