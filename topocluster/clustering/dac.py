"""Model that contains all."""
from typing import Iterator, Optional, Tuple, final

from torch import Tensor
import torch
import torch.nn as nn

from topocluster.clustering.pseudo_labelers import PseudoLabeler
from topocluster.clustering.pseudo_labeling import PlMethod

__all__ = ["DeepClustering"]


@final
class DeepClustering(nn.Module):
    """This class brings everything together into one model object."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        method: PlMethod,
        pseudo_labeler: PseudoLabeler,
        train_encoder: bool,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.method = method
        self.pseudo_labeler = pseudo_labeler
        self.train_encoder = train_encoder

    def supervised_loss(
        self, x: Tensor, class_id: Tensor, ce_weight: float = 1.0, bce_weight: float = 1.0
    ) -> Tuple[Tensor, LoggingDict]:
        return self.method.supervised_loss(
            encoder=self.encoder,
            classifier=self.classifier,
            x=x,
            class_id=class_id,
            ce_weight=ce_weight,
            bce_weight=bce_weight,
        )

    def unsupervised_loss(self, x: Tensor) -> Tuple[Tensor, LoggingDict]:
        return self.method.unsupervised_loss(
            encoder=self.encoder,
            pseudo_labeler=self.pseudo_labeler,
            classifier=self.classifier,
            x=x,
        )

    def step(self, grads: Optional[Tensor] = None) -> None:
        self.classifier.step(grads)
        if self.train_encoder:
            self.encoder.step(grads)

    def zero_grad(self) -> None:
        self.classifier.zero_grad()
        if self.train_encoder:
            self.encoder.zero_grad()

    def train(self) -> None:
        self.classifier.train()
        if self.train_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()

    def eval(self) -> None:
        self.encoder.eval()
        self.classifier.eval()

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))
