from __future__ import annotations
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.tensor import Tensor
from umap import UMAP as _UMAP

from kit import implements


__all__ = ["Reducer", "NoReduce", "UMAP"]


class Reducer(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, X: Tensor, y: Tensor | None = None) -> Reducer:
        ...

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
    def transform(self, X: Tensor, y: Tensor | None = None) -> Tensor:
        return X


class UMAP(_UMAP, Reducer):
    @implements(Reducer)
    def fit(self, X: Tensor, y: Tensor | None) -> UMAP:
        X = X.detach().cpu().numpy()
        if y is not None:
            y = y.detach().cpu().numpy()
        _UMAP.fit(self, X, y)

        return self

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        X = X.detach().cpu().numpy()
        X_transformed = _UMAP.transform(self, X)
        return torch.as_tensor(X_transformed, device=X.device)  # type: ignore
