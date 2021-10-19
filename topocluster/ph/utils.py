from __future__ import annotations

import matplotlib.pyplot as plt
from torch import Tensor

__all__ = ["plot_persistence"]


def plot_persistence(persistence_pairs: Tensor, threshold: float | None = None) -> plt.Figure:
    dgm = persistence_pairs.numpy()
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(dgm[:, 0], dgm[:, 1], s=15, c="red")  # type: ignore

    ax.set_ylabel("Death")
    ax.set_xlabel("Birth")
    lims = ax.get_xlim()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.plot(lims, lims, c="black", alpha=0.6)  # type: ignore
    ax.fill_between(lims, lims, lims[-1], interpolate=True, color="lightgrey")

    title = "Persistence diagram"
    if threshold is not None:
        title += rf" ($\tau={threshold}$)"
    ax.set_title(title)

    return fig
