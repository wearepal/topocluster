from __future__ import annotations
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from torch import Tensor

__all__ = ["plot_persistence"]


def plot_persistence(
    persistence_pairs: Tensor | Sequence | float,
    inf_components: Tensor | Sequence[float],
    threshold: float | None = None,
    inf_delta: float = 0.08,
) -> plt.Figure:
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(persistence_pairs[:, 0], persistence_pairs[:, 1], s=30, alpha=0.5, c="red")  # type: ignore
    lims = list(ax.get_ylim())
    y_min = lims[1]

    if inf_components is not None:
        delta = (lims[1] - lims[0]) * inf_delta
        # Replace infinity values with max_death + delta for diagram to be more
        # readable
        infinity = lims[0] - delta
        y_min = lims[0] - 2 * delta
        ax.scatter(inf_components, infinity, linewidth=1.0, color="green")
        ax.axhline(infinity, ls=":", lw=1, color="k")
        # Infinity label
        yt = ax.get_yticks()
        yt = yt[np.where(yt > infinity)]  # to avoid ploting ticklabel less than -infinity
        yt = np.append(yt, infinity)
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ytl = ["%.3f" % e for e in yt]  # to avoid float precision error
        ytl[-1] = r'$-\infty$'
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    if inf_components is None:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.set_ylabel("Death")
    ax.set_xlabel("Birth")
    ax.set_xlim(lims)
    ax.set_ylim(y_min, lims[1])

    ax.plot(lims, lims, c="black", alpha=0.6)  # type: ignore
    ax.fill_between(lims, lims, lims[-1], interpolate=True, color="lightgrey")

    title = "Persistence diagram"
    if threshold is not None:
        title += rf" ($\tau={threshold}$)"
    ax.set_title(title)

    return fig
