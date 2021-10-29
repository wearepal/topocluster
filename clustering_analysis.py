from __future__ import annotations

if 1:
    import faiss  # type: ignore

from enum import Enum
import math
from pathlib import Path
from typing import Any, Optional

from conduit.models.utils import prefix_keys
import matplotlib.pyplot as plt
import numpy as np
from ranzen.torch.utils import random_seed
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import torch
from torch import Tensor, optim
from tqdm import tqdm
import typer
from umap import UMAP

from topocluster import search
from topocluster.metrics import clustering_accuracy
from topocluster.ph import DTMDensity, cluster_h0, sort_persistence_pairs
from topocluster.ph.utils import plot_persistence
from topocluster.viz import visualize_clusters
import wandb


class WandbMode(Enum):
    """Make W&B either log online, offline or not at all."""

    online = "online"
    offline = "offline"
    disabled = "disabled"


def main(
    ctx: typer.Context,
    k_graph: int = typer.Option(..., "--k-graph", "-kg"),
    greedy: bool = typer.Option(False, "--greedy", "-g"),
    num_samples: int = typer.Option(10_000, "--num-samples", "-ns"),
    k_density: Optional[int] = typer.Option(None, "--k-density", "-kd"),
    q: int = typer.Option(2, "--q", "-q"),
    tau_min: float = typer.Option(0, "--tau-min", "-tmin"),
    tau_max: float = typer.Option(0, "--tau-max", "-tmax"),
    num_tau: int = typer.Option(1, "--num-tau", "-nt"),
    ft_iters: int = typer.Option(0, "--ft-iters", "-it"),
    n_components: Optional[int] = typer.Option(None, "--n-components", "-nc"),
    wandb_mode: WandbMode = typer.Option("offline", "--wandb-mode", "-wm"),
) -> None:

    wandb.init(
        entity="predictive-analytics-lab",
        project="topocluster",
        mode=wandb_mode.name,
        config=ctx.params,
    )
    random_seed(seed_value=47, use_cuda=False)
    make_viz = wandb_mode is not WandbMode.disabled

    path_to_encodings = Path("post_pretrain_train_encodings_ldim=10.pt")
    data = torch.load(path_to_encodings)
    x = data["encodings"]
    sample_inds = torch.randperm(len(x))[:num_samples]
    x = x[sample_inds]
    y = data["labels"][sample_inds]
    y_np = y.numpy()

    num_classes = len(np.unique(y_np))

    knn_g = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=5)
    knn_d = (
        None if k_density is None else search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=5)
    )

    umap_x = None
    if make_viz:

        typer.echo(f"Reducing data with UMAP.")
        mapper = UMAP(n_neighbors=25, n_components=2)
        umap_x = mapper.fit_transform(x.detach().cpu().numpy())

        assert isinstance(umap_x, np.ndarray)
        fig = visualize_clusters(
            umap_x,
            labels=y_np,
            title=f"UMAP Projection (step=0)",
            legend=True,
            top_k=None,
            palette="bright",
        )
        wandb.log({"umap": wandb.Image(fig)})
        plt.close(fig)
    else:
        umap_x = None

    def get_clustering_inputs(_q: float = 2.0, verbose: bool = True) -> tuple[Tensor, Tensor]:
        if verbose:
            typer.echo("Computing the neighborhood graph.")
        knn_out = knn_g(x, return_distances=True)
        graph = knn_out.indices
        if verbose:
            typer.echo("Computing the density map.")
        if knn_d is not None:
            knn_out = knn_d(x, return_distances=True)

        density_map = DTMDensity.from_dists(knn_out.distances, dim=x.size(1), normalize=False, q=_q)
        density_map = density_map.log()
        assert not torch.any(density_map.isnan())
        return graph, density_map

    # fine-tuning
    if ft_iters > 0:
        n_components = num_classes if n_components is None else n_components
        x.requires_grad_(True)
        optimizer = optim.AdamW([x], lr=1.0e-2)
        with tqdm(desc="fine-tuning representation", total=ft_iters) as pbar:
            # destnum_schedule = [20] * 10 + [15] * 10 + [10] * (max(0, ft_iters - 20))
            for step in range(ft_iters):
                graph, density_map = get_clustering_inputs(_q=q, verbose=False)
                merge_out = cluster_h0(
                    neighbor_graph=graph,
                    density_map=density_map,
                    threshold=float("inf"),
                    greedy=greedy,
                )
                pers_pairs = sort_persistence_pairs(
                    merge_out.persistence_pairs, density_map=density_map
                )

                shrinkage_loss = pers_pairs.persistence[n_components - 1 :].sum()
                saliency_loss = -pers_pairs.persistence[: n_components - 1].sum()
                loss = shrinkage_loss + saliency_loss
                pbar.set_postfix(loss=loss.item(), n_components=len(pers_pairs))
                loss.backward()
                optimizer.step()
                pbar.update()

                logging_dict: dict[str, Any] = {
                    "loss/saliency": saliency_loss.item(),
                    "loss/shrinkage": shrinkage_loss.item(),
                    "n_components": len(pers_pairs),
                }

                if make_viz and (((step + 1) % 5) == 0):
                    # typer.echo(f"Reducing data with UMAP.")
                    mapper = UMAP(n_neighbors=25, n_components=2)
                    umap_x = mapper.fit_transform(x.detach().cpu().numpy())
                    assert isinstance(umap_x, np.ndarray)

                    components = pers_pairs.components
                    fig = visualize_clusters(
                        umap_x,
                        labels=y_np,
                        title=f"UMAP Projection (step={step+1}, k={k_graph})",
                        legend=True,
                        top_k=components,
                        palette="bright",
                    )
                    logging_dict["umap"] = wandb.Image(fig)
                    plt.close(fig)

                    fig, ax = plt.subplots(dpi=100)
                    ax.plot(
                        range(len(pers_pairs.inf_components) + 1, len(components) + 1),
                        pers_pairs.persistence.detach().cpu().numpy(),
                    )
                    ax.set_ylabel("Persistence")
                    ax.set_xlabel("Rank")
                    ax.set_title("Persistence vs. Rank")

                    logging_dict["persistence"] = wandb.Image(fig)
                    plt.close(fig)

                logging_dict = prefix_keys(logging_dict, prefix="ft")
                wandb.log(logging_dict)

    graph, density_map = get_clustering_inputs(_q=q)
    taus = np.linspace(tau_min, tau_max, num_tau).tolist()
    taus.append(float("inf"))

    one_nn = search.KnnExact(k=1, normalize=False)
    for tau in taus:
        # for tau in [float("inf")]:
        typer.echo(f"\nClustering on {len(x)} data-points with threshold={tau}.")
        merge_out = cluster_h0(
            neighbor_graph=graph, density_map=density_map, threshold=tau, greedy=greedy
        )
        labels = merge_out.root_idxs
        centers = np.unique(merge_out.root_idxs)
        labels2 = centers[one_nn(x=x, y=x[centers])].squeeze()
        typer.echo(
            "Fraction of labels shared by k-center and neighbourhood cluster-assignment strategies: "
            f"{(labels == labels2).mean()}"
        )

        num_clusters = len(np.unique(labels))
        typer.echo(f"Number of clusters: {num_clusters}")

        ami = adjusted_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"AMI: {ami}")
        nmi = normalized_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"NMI: {nmi}")
        acc = clustering_accuracy(labels_true=y, labels_pred=labels)
        typer.echo(f"Accuracy: {acc}%")

        pers_pairs = None
        if math.isinf(tau):
            pers_pairs = sort_persistence_pairs(
                merge_out.persistence_pairs, density_map=density_map
            )
            plot_persistence(
                density_map[pers_pairs.indices], density_map[pers_pairs.inf_components]
            )
            # plt.show()
            components = pers_pairs.components

            typer.echo(f"Number of components: {len(pers_pairs)}")
            typer.echo(f"Number of finite components: {len(pers_pairs.indices)}")
            typer.echo(f"Number of infinite components: {len(pers_pairs.inf_components)}")

        if make_viz:
            if pers_pairs is not None:
                fig, ax = plt.subplots(dpi=100)
                plt.style.use("seaborn-bright")
                ax.plot(
                    range(len(pers_pairs.inf_components) + 1, len(components) + 1),
                    pers_pairs.persistence,
                )
                ax.set_ylabel("Persistence")
                ax.set_xlabel("Rank")
                ax.set_title(rf"Persistence vs. Rank (k={k_graph}, $\tau={tau}$)")
                # plt.show()

            if umap_x is not None:
                fig = visualize_clusters(
                    umap_x,
                    labels=y_np,
                    title=rf"$\tau={tau}, k={k_graph}$",
                    legend=True,
                    top_k=None if pers_pairs is None else pers_pairs.components,
                    palette="bright",
                )
                # plt.show()
                plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
