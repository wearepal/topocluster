from __future__ import annotations
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional

import faiss  # type: ignore
from gudhi.clustering.tomato import Tomato
import matplotlib.pyplot as plt
import numpy as np
from ranzen.torch.utils import random_seed
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import torch
import typer

from topocluster import search
from topocluster.metrics import clustering_accuracy
from topocluster.ph import DTMDensity, merge_h0
from topocluster.viz import visualize_clusters, visualize_merging


class Method(Enum):
    h0 = "h0"
    tomato = "tomato"


def main(
    k_graph: int = typer.Option(..., "--k-graph", "-kg"),
    method: Method = typer.Option("h0", "--method", "-m"),
    save_dir: Optional[Path] = typer.Option(None, "--save-dir", "-s"),
    num_samples: int = typer.Option(10_000, "--num-samples", "-n"),
    k_density: Optional[int] = typer.Option(None, "--k-density", "-kd"),
    tau_min: float = typer.Option(0, "--tau-min", "-tmin"),
    tau_max: float = typer.Option(5, "--tau-max", "-tmax"),
    num_tau: int = typer.Option(15, "--num-tau", "-nt"),
) -> None:
    random_seed(seed_value=42, use_cuda=False)
    path_to_encodings = Path("post_pretrain_train_encodings_ldim=10.pt")
    data = torch.load(path_to_encodings)
    x = data["encodings"]
    sample_inds = torch.randperm(len(x))[:num_samples]
    x = x[sample_inds]
    y = data["labels"][sample_inds]
    y_np = y.numpy()

    typer.echo("Computing the neighborhood graph.")
    knn = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=100)
    knn_out = knn(x, return_distances=True)
    graph = knn_out.indices

    typer.echo("Computing the density map.")
    if (k_density is not None) and (k_density != k_graph):
        knn = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=100)
        knn_out = knn(x, return_distances=True)
    density_map = DTMDensity.from_dists(knn_out.distances, dim=x.size(1), normalize=False)
    density_map = density_map.log()
    assert not torch.any(density_map.isnan())

    if save_dir is not None:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        pred_dir = save_dir / "predicted"
        pred_dir.mkdir(parents=True)
        typer.echo(f"Save directory created at {str(save_dir.resolve())}")

        from umap import UMAP

        typer.echo(f"Reducing data with UMAP.")
        mapper = UMAP(n_neighbors=15, n_components=2)
        umap_x = mapper.fit_transform(x)
        assert isinstance(umap_x, np.ndarray)
        gt_viz = visualize_clusters(umap_x, labels=y_np, title="Ground Truth", legend=True)
        gt_viz.savefig(save_dir / f"ground_truth.png")
    else:
        pred_dir = None
        umap_x = None

    for i, tau in enumerate(np.linspace(tau_min, tau_max, num_tau)):
        typer.echo(f"\nClustering on {len(x)} data-points with threshold={tau}.")
        if method is Method.h0:
            merge_out = merge_h0(
                neighbor_graph=graph, density_map=density_map, threshold=tau, store_tree=True
            )
            labels = merge_out.labels
        else:
            tomato = Tomato(weights_=density_map.numpy(), merge_threshold=tau)
            tomato.fit(x.numpy())
            labels = tomato.labels_

        # print(merge_out.labels.unique())
        ami = adjusted_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"AMI: {ami}")
        nmi = normalized_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"NMI: {nmi}")
        typer.echo(f"Number of clusters: {len(np.unique(labels))}")
        acc = clustering_accuracy(labels_true=y, labels_pred=labels)
        typer.echo(f"Accuracy: {acc}%")

        # # pd = plot_persistence(merge_out.barcode, threshold=tau)
        if (pred_dir is not None) and (umap_x is not None):
            fig = visualize_clusters(umap_x, labels=labels, title=rf"$\tau={tau}$", legend=False)
            fig.savefig(pred_dir / f"{i}.png")
            plt.close(fig)

    if save_dir is not None:
        visualize_merging(save_dir)


if __name__ == "__main__":
    typer.run(main)
