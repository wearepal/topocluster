from __future__ import annotations
from abc import abstractmethod
from typing import Dict

from torch import Tensor

__all__ = ["Clusterer"]


class Clusterer:

    hard_labels: Tensor
    soft_labels: Tensor

    @abstractmethod
    def fit(self, x: Tensor) -> Clusterer:
        ...

    @abstractmethod
    def get_loss(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        ...

    @abstractmethod
    def build(self, input_dim: int, num_classes: int) -> None:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.hard_labels
