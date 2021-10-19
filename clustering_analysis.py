from __future__ import annotations

if 1:
    import faiss  # type: ignore

from enum import Enum
from pathlib import Path
import shutil
from typing import Optional

import gudhi
import matplotlib.pyplot as plt
import numpy as np
from ranzen.torch import prop_random_split
from ranzen.torch.utils import random_seed
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import timm
import torch
from torch import Tensor, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
import torchvision.transforms.functional as TF
from tqdm import tqdm
import typer

from topocluster import search
from topocluster.metrics import clustering_accuracy
from topocluster.ph import DTMDensity, cluster_h0, sort_persistence_pairs
from topocluster.viz import visualize_clusters


class Method(Enum):
    h0 = "h0"
    tomato = "tomato"


def main(
    k_graph: int = typer.Option(..., "--k-graph", "-kg"),
    method: Method = typer.Option("h0", "--method", "-m"),
    save_dir: Optional[Path] = typer.Option(None, "--save-dir", "-s"),
    num_samples: int = typer.Option(10_000, "--num-samples", "-n"),
    k_density: Optional[int] = typer.Option(None, "--k-density", "-kd"),
    q: int = typer.Option(2, "--q", "-q"),
    tau_min: float = typer.Option(0, "--tau-min", "-tmin"),
    tau_max: float = typer.Option(5, "--tau-max", "-tmax"),
    num_tau: int = typer.Option(15, "--num-tau", "-nt"),
    gd_iters: int = typer.Option(0, "--gd-iters", "-it"),
    destnum: Optional[int] = typer.Option(None, "--destnum", "-dn"),
) -> None:
    random_seed(seed_value=47, use_cuda=False)
    # mnist = MNIST(root="data", train=True, download=True)
    # # model: ResNetV2 = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    # # model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    # # model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # data = TF.normalize(
    #     (mnist.data / 255.0).unsqueeze(1).expand(-1, 3, -1, -1),
    #     mean=mean,
    #     std=std,
    # )
    # ds = TensorDataset(data, mnist.targets)
    # props = min(num_samples / len(ds), 1.0)
    # ds = prop_random_split(ds, props=props)
    # if isinstance(ds, list):
    #     ds = ds[0]

    # dl = DataLoader(ds, batch_size=256, shuffle=False)
    # del mnist

    # x_ls = []
    # y_ls = []
    # with torch.no_grad():
    #     for x, y in dl:
    #         # intermediate_output = model.get_intermediate_layers(x, 4)
    #         # x_enc = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
    #         # x = K.resize(x, size=(224, 224))
    #         x_enc = model(x)
    #         x_ls.append(x_enc)
    #         y_ls.append(y)
    # x = torch.cat(x_ls, dim=0)
    # y = torch.cat(y_ls, dim=0)
    # y_np = y.numpy()
    # del ds

    # x = F.normalize(x, dim=1, p=2)

    path_to_encodings = Path("post_pretrain_train_encodings_ldim=10.pt")
    data = torch.load(path_to_encodings)
    x = data["encodings"]
    sample_inds = torch.randperm(len(x))[:num_samples]
    x = x[sample_inds]
    y = data["labels"][sample_inds]
    y_np = y.numpy()

    num_classes = len(np.unique(y_np))

    knn_g = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=5)
    knn_d = None

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

    # topograd
    if gd_iters > 0:
        destnum = num_classes if destnum is None else destnum
        x.requires_grad_(True)
        optimizer = optim.AdamW([x], lr=1.0e-2)
        with tqdm(desc="topograd", total=gd_iters) as pbar:
            for i in range(gd_iters):
                # typer.echo(f"Iteration {i}/{gd_iters} of topograd.")
                graph, density_map = get_clustering_inputs(_q=q, verbose=False)
                merge_out = cluster_h0(
                    neighbor_graph=graph,
                    density_map=density_map,
                    threshold=float("inf"),
                    greedy=method is Method.h0,
                )
                pers_pairs = sort_persistence_pairs(
                    merge_out.persistence_pairs, density_map=density_map
                )

                loss = pers_pairs.persistence[destnum - 1 :].sum()
                loss -= pers_pairs.persistence[: destnum - 1].sum()
                pbar.set_postfix(loss=loss.item(), n_components=len(pers_pairs))
                loss.backward()
                optimizer.step()
                pbar.update()

    graph, density_map = get_clustering_inputs(_q=q)

    if save_dir is not None:
        from umap import UMAP

        typer.echo(f"Reducing data with UMAP.")
        mapper = UMAP(n_neighbors=25, n_components=2)
        umap_x = mapper.fit_transform(x.detach().cpu().numpy())
        assert isinstance(umap_x, np.ndarray)

        pred_dir = save_dir / "predicted"
        pred_dir.mkdir(parents=True, exist_ok=True)

        gt_viz = visualize_clusters(
            umap_x,
            labels=y_np,
            title=f"Ground Truth",
            # title=f"Density Map (kd={k_density}, q={q})",
            legend=True,
            top_k=None,
            palette="bright",
        )
        # gt_viz.savefig(save_dir / f"kviz={len(top_k)}_kd={knn_d}_ground_truth.png")
        gt_viz.savefig(save_dir / "ground_truth.png")
        plt.show()
        plt.close(gt_viz)
    else:
        pred_dir = None
        umap_x = None

    graph, density_map = get_clustering_inputs(_q=2)
    taus = np.linspace(tau_min, tau_max, num_tau).tolist()
    taus.append(float("inf"))
    for i, tau in enumerate(taus):
        # for tau in [float("inf")]:
        typer.echo(f"\nClustering on {len(x)} data-points with threshold={tau}.")
        merge_out = cluster_h0(
            neighbor_graph=graph, density_map=density_map, threshold=tau, greedy=method is Method.h0
        )
        labels = merge_out.labels
        num_clusters = len(np.unique(labels))
        typer.echo(f"Number of clusters: {num_clusters}")
        if num_clusters < num_classes:
            break

        # print(merge_out.labels.unique())
        ami = adjusted_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"AMI: {ami}")
        nmi = normalized_mutual_info_score(labels_true=y, labels_pred=labels)
        typer.echo(f"NMI: {nmi}")
        acc = clustering_accuracy(labels_true=y, labels_pred=labels)
        typer.echo(f"Accuracy: {acc}%")

        # pd = plot_persistence(merge_out.barcode, threshold=tau)
        if (pred_dir is not None) and (umap_x is not None):
            fig = visualize_clusters(
                umap_x,
                labels=y_np,
                title=rf"$\tau={tau}$",
                legend=True,
                # top_k=components[:10],
                palette="bright",
            )
            fig.savefig(pred_dir / f"{i}.png")
            plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
