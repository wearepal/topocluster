from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Tuple

from torch import Tensor

__all__ = ["Clusterer"]


class Clusterer:
    @abstractmethod
    def build(self, input_dim: int, num_classes: int) -> None:
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def get_loss(
        self, x: Tensor, soft_labels: Tensor, hard_labels: Tensor, y: Tensor, prefix: str = ""
    ) -> Dict[str, Tensor]:
        ...

    def routine(self, x: Tensor, y: Tensor, prefix: str = "") -> Dict[str, Tensor]:
        hard_labels, soft_labels = self.forward(x)
        return self.get_loss(
            x=x, soft_labels=soft_labels, hard_labels=hard_labels, y=y, prefix=prefix
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(x)
