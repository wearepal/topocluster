from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
import torch.nn as nn

__all__ = ["Clusterer"]


class Clusterer:

    hard_labels: Tensor
    soft_labels: Tensor

    @abstractmethod
    def fit(self, x: Tensor) -> Clusterer:
        ...

    @abstractmethod
    def get_loss(self, x: Tensor) -> Optional[Tensor]:
        ...

    @abstractmethod
    def build(self, input_dim: int, num_classes: int) -> None:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.hard_labels
