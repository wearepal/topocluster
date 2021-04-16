from __future__ import annotations
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
import torch
from torch.tensor import Tensor
from umap import UMAP as _UMAP

from kit import implements


__all__ = ["Reducer", "NoReduce", "UMAP", "RandomProjector"]


class Reducer(BaseEstimator, ABC):
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
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.proj_dim = proj_dim

    @implements(Reducer)
    def transform(self, X: Tensor) -> Tensor:
        proj_matrix = torch.normal(
            mean=0, std=1 / self.proj_dim, size=(X.shape[1], self.proj_dim), device=X.device
        )
        return X @ proj_matrix
