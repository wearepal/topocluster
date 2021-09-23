import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

__all__ = ["plot_persistence"]


def plot_persistence(persistence_pairs: Tensor) -> plt.Figure:
    barcodes_np = persistence_pairs.numpy() / persistence_pairs.max()
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(barcodes_np[:, 0], barcodes_np[:, 1], s=15, c="red")  # type: ignore
    span = np.array([0, 1])
    ax.plot(span, span, c="black", alpha=0.6)  # type: ignore
    ax.fill_between(span, span, 0, interpolate=True, color="lightgrey")
    ax.set_ylabel("Death")
    ax.set_xlabel("Birth")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Persistence diagram")

    return fig
