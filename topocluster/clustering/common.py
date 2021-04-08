from __future__ import annotations
from abc import abstractmethod

from torch import Tensor

from topocluster.data.datamodules import DataModule
from topocluster.models.base import Encoder

__all__ = ["Clusterer"]


class Clusterer:
    @abstractmethod
    def build(self, encoder: Encoder, datamodule: DataModule) -> None:
        ...

    @abstractmethod
    def _get_loss(self, x: Tensor) -> dict[str, Tensor]:
        ...

    def get_loss(self, x: Tensor, prefix: str = "") -> dict[str, Tensor]:
        if prefix:
            prefix += "/"
        # Prepend the prefix to all keys of the loss dict
        loss_dict = self._get_loss(x=x)
        return {prefix + key: value for key, value in loss_dict.items()}

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...
