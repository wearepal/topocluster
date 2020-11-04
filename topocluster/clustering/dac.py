"""Model that contains all."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Tuple, final

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from topocluster.clustering.common import Clusterer
from topocluster.data.data_modules import MASK_VALUE
from topocluster.utils.torch_ops import dot_product, normalized_softmax

__all__ = ["PlClusterer"]


class PlClusterer(Clusterer, nn.Module):
    classifier: nn.Linear

    def fit(self, x: Tensor) -> PlClusterer:
        self.logits = self.classifier(x)
        self.labels = self.logits.argmax(dim=-1)
        return self

    def build(self, input_dim: int, num_classes: int) -> None:
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return Clusterer.__call__(self, x)


def cosine_and_bce(probs: Tensor, labels: Tensor) -> Tensor:
    """Cosine similarity and then binary cross entropy."""
    # cosine similarity
    mask = labels == MASK_VALUE
    probs = probs[mask]
    labels = labels[mask]

    cosine_sim = dot_product(probs[:, None, :], probs).clamp(min=0, max=1)
    # binary cross entropy
    unreduced_loss = F.binary_cross_entropy(cosine_sim, labels, reduction="none")
    return torch.mean(unreduced_loss * mask)


class PseudoLabelLoss(nn.Module, ABC):
    def __init__(self, pseudo_labeler: Callable[[Tensor], Tensor]) -> None:
        self.pseudo_labeler = pseudo_labeler

    @abstractmethod
    def forward(self, encodings: Tensor, logits: Tensor) -> Tensor:
        ...


class PlCrossEntropy(PseudoLabelLoss):
    def forward(self, encoding: Tensor, logits: Tensor) -> Tensor:
        pseudo_labels = self.pseudo_labeler(encoding)
        return F.cross_entropy(logits, pseudo_labels)


class PlCosineBCE(PseudoLabelLoss):
    """Cosine-BCE loss."""

    def forward(self, encoding: Tensor, logits: Tensor) -> Tensor:
        # only do softmax but no real normalization
        probs = logits.softmax(dim=-1)
        pseudo_labels = self.pseudo_labeler(encoding)  # base the pseudo labels on the encoding
        return cosine_and_bce(probs, pseudo_labels)


class PlNormalizedCosineBCE(PseudoLabelLoss):
    """Normalize the probabilities by the l2 norm before computing the Cosine-BCE loss."""

    def forward(self, encoding: Tensor, logits: Tensor) -> Tensor:
        # only do softmax but no real normalization
        probs = normalized_softmax(logits)
        pseudo_labels = self.pseudo_labeler(encoding)  # base the pseudo labels on the encoding
        return cosine_and_bce(probs, pseudo_labels)
