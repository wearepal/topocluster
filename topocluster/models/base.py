from __future__ import annotations
from abc import abstractmethod
from typing import cast

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.tensor import Tensor

from kit import implements
from topocluster.data.datamodules import DataModule
from topocluster.data.utils import Batch


__all__ = ["Encoder"]


class Encoder(pl.LightningModule):
    """Base class for AutoEncoder models."""

    encoder: nn.Module

    def __init__(self, lr: float = 1.0e-3) -> None:
        super().__init__()
        self.lr = lr

    @abstractmethod
    def _build(self, datamodule: DataModule) -> nn.Module:
        ...

    def build(self, datamodule: DataModule) -> None:
        self.encoder = self._build(datamodule)

    @implements(nn.Module)
    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    @abstractmethod
    def _get_loss(self, encoding: Tensor, batch: Batch, prefix: str = "") -> dict[str, Tensor]:
        ...

    def get_loss(self, encoding: Tensor, batch: Batch, prefix: str = "") -> dict[str, Tensor]:
        if prefix:
            prefix += "/"
        loss_dict = self._get_loss(encoding=encoding, batch=batch)
        # Prepend the prefix to all keys of the loss dict
        return {prefix + key: value for key, value in loss_dict.items()}

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x, _ = batch
        encoding = self.encoder(x)
        loss_dict = self.get_loss(encoding, x, prefix="train/")
        total_loss = cast(Tensor, sum(loss_dict.values()))
        self.log_dict(loss_dict)
        return total_loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, Tensor]:
        x, _ = batch
        encoding = self.encoder(x)
        loss_dict = self.get_loss(encoding, batch=batch, prefix="val/")
        self.log_dict(loss_dict)
        return loss_dict