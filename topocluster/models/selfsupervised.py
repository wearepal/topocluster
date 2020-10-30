from abc import abstractmethod

import pytorch_lightning as pl
import torch
from torch import Tensor, nn


__all__ = ["SelfSupervised"]


class SelfSupervised(pl.LightningModule):
    """Encoder trained with self-supervision."""

    def __init__(self, encoder: nn.Module, classifier: nn.Module, lr: float = 1.0e-3):
        self.save_hyperparameters()
        self.encoder = encoder
        self.classifier = classifier

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    @abstractmethod
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        ...
