from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

__all__ = ["plot_persistence"]


def plot_persistence(persistence_pairs: Tensor, threshold: float | None = None) -> plt.Figure:
    barcodes_np = persistence_pairs.numpy()
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(barcodes_np[:, 0], barcodes_np[:, 1], s=15, c="red")  # type: ignore

    ax.set_ylabel("Death")
    ax.set_xlabel("Birth")
    x_lims = [0, persistence_pairs[:, 0].max(dim=0).values.item()]
    ax.set_xlim(x_lims)
    y_lims = [0, persistence_pairs[:, 1].max(dim=0).values.item()]
    ax.set_ylim(y_lims)

    ax.plot(x_lims, y_lims, c="black", alpha=0.6)  # type: ignore
    ax.fill_between(x_lims, y_lims, 0, interpolate=True, color="lightgrey")

    title = "Persistence diagram"
    if threshold is not None:
        title += rf" ($\tau={threshold}$)"
    ax.set_title(title)

    return fig
