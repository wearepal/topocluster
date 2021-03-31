from __future__ import annotations
from typing import ClassVar

from torch import Tensor
import torch
import torch.nn as nn

from topocluster.data.datamodules import VisionDataModule
from topocluster.data.utils import Batch
from topocluster.models.base import Encoder


class LeNet4(Encoder):
    criterion: ClassVar[nn.loss._Loss]
    """

    Adapted from https://github.com/activatedgeek/LeNet-5
    """

    def _build(self, datamodule: VisionDataModule) -> nn.Module:

        encoder = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.LazyConv2d(16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.LazyConv2d(120, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Flatten(),
        )
        logits_dim = datamodule.num_classes if datamodule.num_classes > 2 else 1
        self.fc = nn.LazyLinear(logits_dim)
        self.criterion = nn.CrossEntropyLoss() if logits_dim > 2 else nn.BCEWithLogitsLoss()
        # Lazy initialization
        self.fc(encoder(torch.ones(datamodule.dims)[None]))
        return encoder

    def _get_loss(self, encoding: Tensor, batch: Batch) -> dict[str, Tensor]:
        logits = self.fc(encoding)
        targets = batch.y
        if logits.size(1) == 1:
            logits = logits.flatten()
            targets = targets.flatten().float()
        else:
            targets = targets.long()
        return {"classification_loss": self.criterion(input=logits, target=targets)}
