from __future__ import annotations
from enum import Enum, auto
import time
from typing import Optional

import faiss
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder

__all__ = ["GMM"]


class GMM(Clusterer, GaussianMixture):
    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.n_components = datamodule.num_classes * datamodule.num_subgroups

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_np = x.detach().cpu().numpy()
        hard_labels = torch.as_tensor(self.fit_predict(x_np))
        soft_labels = torch.as_tensor(self.predict_proba(x_np))

        return hard_labels, soft_labels

    def get_loss(self, x: Tensor) -> dict[str, Tensor]:
        return {}
