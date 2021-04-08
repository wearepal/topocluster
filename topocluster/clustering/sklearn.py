from __future__ import annotations

from sklearn.cluster import (
    AgglomerativeClustering as _AgglomerativeClustering,
    OPTICS as _OPTICS,
    SpectralClustering as _SpectralClustering,
)
from sklearn.mixture import GaussianMixture as _GaussianMixture
import torch
from torch import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder

__all__ = ["GaussianMixture", "AgglomerativeClustering", "SpectralClustering"]


class GaussianMixture(Clusterer, _GaussianMixture):
    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.n_components = datamodule.num_classes * datamodule.num_subgroups

    def __call__(self, x: Tensor) -> Tensor:
        x_np = x.detach().cpu().numpy()
        return torch.as_tensor(self.fit_predict(x_np))

    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        return {}


class AgglomerativeClustering(Clusterer, _AgglomerativeClustering):
    def __call__(self, x: Tensor) -> Tensor:
        x_np = x.detach().cpu().numpy()
        return torch.as_tensor(self.fit_predict(x_np))

    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.n_clusters = datamodule.num_classes * datamodule.num_subgroups

    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        return {}


class SpectralClustering(Clusterer, _SpectralClustering):
    def __call__(self, x: Tensor) -> Tensor:
        x_np = x.detach().cpu().numpy()
        return torch.as_tensor(self.fit_predict(x_np))

    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        self.n_clusters = datamodule.num_classes * datamodule.num_subgroups

    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        return {}
