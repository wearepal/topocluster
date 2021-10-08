from __future__ import annotations
from typing import ClassVar

import attr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from torch import Tensor

__all__ = ["ElevenNodes"]


class ElevenNodes:
    THRESHOLDS: ClassVar[tuple[float, ...]] = (0, 0.31, 0.51, 0.56)

    def __init__(self) -> None:
        @attr.define(kw_only=True)
        class Node:
            edges: list[int]
            density: float

        self.nodes = [
            # red
            Node(edges=[1, 2], density=0.85),
            Node(edges=[0, 2], density=0.85),
            Node(edges=[0, 1, 3], density=0.85),
            # yellow
            Node(edges=[2, 4], density=0.5),
            # green
            Node(edges=[3, 5, 6], density=0.8),
            Node(edges=[4, 6], density=0.8),
            Node(edges=[3, 4, 5], density=0.8),
            # purple
            Node(edges=[2, 6, 8], density=0.3),
            # blue
            Node(edges=[7, 9, 10], density=0.9),
            Node(edges=[8, 10], density=0.9),
            Node(edges=[8, 9], density=0.9),
        ]

        self.G = nx.DiGraph()
        density_map_ls = []
        edges = []
        for i, v in enumerate(self.nodes):
            edges.append(torch.as_tensor(v.edges))
            density_map_ls.append(v.density)
            for nbr in v.edges:
                weight = round(v.density - self.nodes[nbr].density, 2)
                self.G.add_edge(nbr, i, weight=weight)

        self.edges = edges
        self.density_map = torch.as_tensor(density_map_ls)
        self._NODE_LABELS = dict(zip(range(len(self.G.nodes)), density_map_ls))
        self._CMAP = np.array(sns.color_palette("Set3", len(self.density_map)).as_hex())  # type: ignore

    def draw_graph(
        self,
        labels: list[int] | Tensor | None,
        title: str = "",
    ) -> plt.Figure:  # type: ignore
        fig, ax = plt.subplots(dpi=100, figsize=(8, 6))
        pos = nx.nx_agraph.graphviz_layout(self.G)
        node_color = self._CMAP if labels is None else self._CMAP[labels]
        nx.draw(
            self.G,
            pos=pos,
            with_labels=False,
            node_color=node_color,
            font_weight="bold",
            connectionstyle='arc3, rad = 0.2',
            node_size=600,
        )
        ax.set_title(title)

        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=edge_labels, font_size=6)
        nx.draw_networkx_labels(
            self.G,
            pos=pos,
            labels=self._NODE_LABELS,
            verticalalignment="center",
            horizontalalignment="center",
            font_size=10,
        )
        return fig

    def __len__(self) -> int:
        return len(self.density_map)
