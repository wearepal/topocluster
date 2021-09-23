from __future__ import annotations

import logging

import attr
import numpy as np
import torch
from kit import implements
from torch import Tensor
from tqdm import tqdm

from topocluster.clustering.common import Clusterer
from topocluster.ph.clustering import Tomato, compute_density_map, compute_rips, tomato
from zero_dim_ph import zero_dim_merge

__all__ = ["topograd_loss", "TopoGrad"]

LOGGER = logging.getLogger(__name__)


@attr.define(kw_only=True)
class TopoGradLoss:
    shrinking: Tensor
    saliency: Tensor

    @property
    def total(self) -> Tensor:
        return self.shrinking + self.saliency


def topograd_loss(pc: Tensor, k_kde: int, k_rips: int, scale: float, destnum: int) -> TopoGradLoss:
    kde_dists, _ = compute_density_map(pc, k_kde, scale)
    sorted_idxs = torch.argsort(kde_dists, descending=False)
    kde_dists_sorted = kde_dists[sorted_idxs]
    pc_sorted = pc[sorted_idxs]
    rips_idxs = compute_rips(pc_sorted, k_rips)

    _, pers_pairs = zero_dim_merge(
        density_map=kde_dists,
        neighbor_graph=rips_idxs,
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
        shrinking_loss = saliency_loss = pc.new_zeros(())
    else:
        # Dimension 1 encodes birth-death (in that order) pairs
        pd_pairs = torch.as_tensor(pd_pairs, device=pc.device)
        oripd = kde_dists_sorted[pd_pairs]
        pers_idxs_sorted = torch.argsort(oripd[:, 0] - oripd[:, 1])

        changing = pers_idxs_sorted[:-destnum]
        nochanging = pers_idxs_sorted[-destnum:]

        changepairs = pd_pairs[changing]
        nochangepairs = pd_pairs[nochanging]
        # shrinking loss is the sum of squares of the distances to the diagonal
        # of the points in the diagram
        shrinking_loss = ((0.5 * kde_dists_sorted[changepairs].diff(dim=1)) ** 2).sum()
        # Our saliency loss is the opposite of the sum of squares of the distances to the diagonal
        # of the points in the diagram
        saliency_loss = -((0.5 * kde_dists_sorted[nochangepairs].diff(dim=1)) ** 2).sum()

    return TopoGradLoss(shrinking=shrinking_loss, saliency=saliency_loss)


class TopoGrad(Tomato):
    def __init__(
        self,
        *,
        k_kde: int,
        k_rips: int,
        destnum: int,
        scale: float,
        threshold: float,
        n_iter: int = 0,
        lr: float = 1e-3,
        sal_loss_w: float = 1.0,
        shrink_loss_w: float = 1.0,
    ) -> None:
        super().__init__(k_kde=k_kde, k_rips=k_rips, scale=scale, threshold=threshold)
        self.n_iter = n_iter
        self.optimizer_cls = torch.optim.AdamW
        self.lr = lr
        self.sal_loss_w = sal_loss_w
        self.shrink_loss_w = shrink_loss_w
        self.destnum = destnum

    def _get_loss(self, x: Tensor) -> TopoGradLoss:
        loss = topograd_loss(
            pc=x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, destnum=self.destnum
        )
        loss.saliency *= self.sal_loss_w
        loss.shrinking *= self.shrink_loss_w
        return loss

    @implements(Clusterer)
    def __call__(self, x: Tensor, *, threshold: float | None = None) -> Tensor:
        threshold = self.threshold if threshold is None else threshold
        # Run topograd on the embedding (without backpropagating through the network)
        if self.n_iter > 0:
            # Avoid modifying the original embedding
            x = x.detach()
            x = x.clone().requires_grad_(True)
            optimizer = self.optimizer_cls((x,), lr=self.lr)
            with tqdm(desc="topograd", total=self.n_iter) as pbar:
                for _ in range(self.n_iter):
                    loss = self._get_loss(x=x).total
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

        output = tomato(
            x, k_kde=self.k_kde, k_rips=self.k_rips, scale=self.scale, threshold=threshold
        )

        self.persistence_pairs = output.persistence_pairs

        return output.cluster_ids
