"""Labelers"""

import torch
from torch import Tensor
import torch.nn as nn

from topocluster.data.data_modules import MASK_VALUE
from topocluster.utils import dot_product

__all__ = ["RankingStatistics", "CosineSimThreshold"]


class RankingStatistics(nn.Module):
    def __init__(self, k_num: int):
        super().__init__()
        self.k_num = k_num

    def _get_topk(self, z: Tensor) -> Tensor:
        return torch.sort(torch.topk(torch.flatten(z, 1), k=self.k_num).indices).values

    def forward(self, z: Tensor) -> Tensor:
        topk_z = self._get_topk(z.abs())
        labels = (topk_z == topk_z[:, None, :]).all(dim=-1).float()
        return labels


class CosineSimThreshold(Tensor):
    def __init__(self, upper_threshold: float, lower_threshold: float):
        super().__init__()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def forward(self, z: Tensor) -> Tensor:
        z = z.flatten(start_dim=1)
        cosine_sim = dot_product(z, z[:, None, :])
        over = (cosine_sim > self.upper_threshold).float()
        under = (cosine_sim < self.lower_threshold).float()
        mask = over + under
        labels = over  # this will ensure that the label for all samples in `under` is 0
        labels[mask] = MASK_VALUE
        return labels
