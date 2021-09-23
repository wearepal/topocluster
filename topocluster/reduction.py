from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from kit import implements
from sklearn.base import BaseEstimator
from sklearn.manifold import TSNE as _TSNE
from torch import Tensor
from umap import UMAP as _UMAP

__all__ = ["Reducer", "NoReduce", "UMAP", "RandomProjector", "TSNE"]


class Reducer(BaseEstimator, ABC):
    n_components: int

    def fit(self, X: Tensor, y: Tensor | None = None) -> Reducer:
        return self

    @abstractmethod
    def transform(self, X: Tensor, y: Tensor | None = None) -> Tensor:
        ...

    def fit_transform(self, X: Tensor, y: Tensor | None = None) -> Tensor:
        self.fit(X, y)
        return self.transform(X)


class NoReduce(Reducer):
    @implements(Reducer)
    def fit(self, X: Tensor, y: Tensor | None) -> Reducer:
        return self

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        return X


class UMAP(Reducer, _UMAP):
    @implements(Reducer)
    def fit(self, X: Tensor, y: Tensor | None) -> UMAP:
        X = X.detach().cpu().numpy()
        if y is not None:
            y = y.detach().cpu().numpy()
        _UMAP.fit(self, X, y)

        return self

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        X_transformed = _UMAP.transform(self, X_np)
        return torch.as_tensor(X_transformed, device=X.device)  # type: ignore


class RandomProjector(Reducer):
    def __init__(self, n_components: int) -> None:
        super().__init__()
        self.n_components = n_components

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        proj_matrix = torch.normal(
            mean=0, std=1 / self.n_components, size=(X.shape[1], self.n_components), device=X.device
        )

        return X @ proj_matrix


class TSNE(Reducer, _TSNE):
    def fit(self, X: Tensor, y: Tensor | None) -> TSNE:
        X = X.detach().cpu().numpy()
        if y is not None:
            y = y.detach().cpu().numpy()
        self.embedding_ = torch.as_tensor(_TSNE._fit(self, X, y), device=X.device)
        return self

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        return torch.as_tensor(self.embedding_, device=X.device)  # type: ignore
