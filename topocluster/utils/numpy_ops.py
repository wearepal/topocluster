from __future__ import annotations
from typing import Tuple

from faiss import IndexFlatL2
import numpy as np

__all__ = ["rbf", "compute_rips", "compute_density_map"]


def rbf(x: np.ndarray, y: np.ndarray, scale: float, axis: int = -1) -> np.ndarray[np.float32]:
    "Compute the distance between two vectors using an RBF kernel."
    return np.exp(-np.linalg.norm(x - y, axis=axis) ** 2 / scale)


def compute_rips(pc: np.ndarray[np.float32], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """"Compute the delta-Rips Graph."""
    pc = pc.astype(np.float32)
    cpuindex = IndexFlatL2(pc.shape[1])
    cpuindex.add(pc)

    return cpuindex.search(pc, k)


def compute_density_map(x: np.ndarray, k: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k-nearest neighbours kernel density estimate."""
    x = x.astype(np.float32)
    index = IndexFlatL2(x.shape[1])
    index.add(x)
    values, indexes = index.search(x, k)
    result = np.sum(np.exp(-values / scale), axis=1) / (k * scale)
    return result / max(result), indexes
