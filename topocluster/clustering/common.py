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
    def get_loss(
        self, x: Tensor, soft_labels: Tensor, hard_labels: Tensor, y: Tensor, prefix: str = ""
    ) -> Dict[str, Tensor]:
        ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ...
