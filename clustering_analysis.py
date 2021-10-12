from __future__ import annotations

if 1:
    import faiss  # type: ignore
import torch.nn.functional as F
from gudhi.wasserstein import wasserstein_distance
from torch import Tensor
from gudhi.weighted_rips_complex import WeightedRipsComplex
from scipy.spatial.distance import cdist
import gudhi
from torchvision.datasets import MNIST
from ranzen.torch import prop_random_split
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset
from enum import Enum
from pathlib import Path
import shutil
from typing import Optional, Sequence, NamedTuple

from gudhi.clustering.tomato import Tomato
import matplotlib.pyplot as plt
import numpy as np
from ranzen.torch.utils import random_seed
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import torch
from torch import optim
import typer

from topocluster import search
from topocluster.metrics import clustering_accuracy
from topocluster.ph import DTMDensity, DTM, merge_h0
from topocluster.viz import visualize_clusters, visualize_merging

import timm




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
    gd_iters: int = typer.Option(0, "--gd-iters", "-it"),
) -> None:
    random_seed(seed_value=42, use_cuda=False)
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
    # ds, _ = prop_random_split(ds, props=props)
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

    typer.echo("Computing the neighborhood graph.")
    knn_g = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=100)

    typer.echo("Computing the density map.")
    knn_d = None
    if (k_density is not None) and (k_density != k_graph):
        knn_d = search.KnnIVF(k=k_graph, normalize=False, nprobe=4, nlist=100)

    def get_clustering_inputs():
        typer.echo("Computing the neighborhood graph.")
        knn_out = knn_g(x, return_distances=True)
        graph = knn_out.indices
        typer.echo("Computing the density map.")
        if knn_d is not None:
            knn_out = knn_d(x, return_distances=True)

        density_map = DTMDensity.from_dists(knn_out.distances, dim=x.size(1), normalize=False)
        density_map = density_map.log()
        assert not torch.any(density_map.isnan())
        return graph, density_map

    # topograd
    # if method is Method.h0 and (gd_iters > 0):
    #     x.requires_grad_(True)
    #     optimizer = optim.AdamW([x], lr=1.0e-2)
    #     for i in range(gd_iters):
    #         typer.echo(f"Iteration {i}/{gd_iters} of topograd.")

    #         complex = DTMRipsComplex(points=x, k=15)
    #         st = complex.create_simplex_tree(max_dimension=1)
    #         st.compute_persistence()

    #         pairs = st.persistence_pairs()
    #         typer.echo(f"Number of clusters: {len(pairs)}")

    #         p1b = torch.tensor([i[0] for i in pairs if (len(i[1]) > 0)])
    #         p1d = torch.tensor([i[1] for i in pairs if (len(i[1]) > 0)])
    #         if (i > 0) and i % 50 == 0:
    #             gudhi.plot_persistence_diagram(st.persistence())
    #             plt.show()
    #         diag = torch.norm(x[p1d] - x[p1b], dim=-1)
    # diag[num_classes] *= -1
    # Total persistence is a special case of Wasserstein distance
    # target_diag = torch.stack(
    #     [torch.zeros(num_classes), torch.full((num_classes,), fill_value=1)], dim=-1
    # )
    # loss = -wasserstein_distance(diag, [], order=2, enable_autodiff=True)

    # graph, density_map = get_clustering_inputs()
    # merge_out = merge_h0(
    #     neighbor_graph=graph, density_map=density_map, threshold=0, store_tree=False
    # )
    # root_idxs = merge_out.root_idxs
    # optimizer.zero_grad()
    # root_idxs_u = root_idxs.unique()
    # modes = density_map[root_idxs_u].sort(descending=True).values
    # loss = (modes[num_classes:] - density_map.min().detach()).mean()
    # loss += density_map.max().detach() - modes[:num_classes].mean()
    # typer.echo(f"Loss: {loss.item()}")
    # # typer.echo(f"Number of clusters: {len(root_idxs_u)}")
    # loss.backward()
    # optimizer.step()

    # x.requires_grad_(False)

    graph, density_map = get_clustering_inputs()

    if save_dir is not None:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        pred_dir = save_dir / "predicted"
        pred_dir.mkdir(parents=True)
        typer.echo(f"Save directory created at {str(save_dir.resolve())}")

        from umap import UMAP

        typer.echo(f"Reducing data with UMAP.")
        mapper = UMAP(n_neighbors=25, n_components=2)
        umap_x = mapper.fit_transform(x)
        assert isinstance(umap_x, np.ndarray)
        gt_viz = visualize_clusters(umap_x, labels=y_np, title="Ground Truth", legend=True)
        # gt_viz.savefig(save_dir / f"ground_truth.png")
        plt.show()
        plt.close(gt_viz)
        return
    else:
        pred_dir = None
        umap_x = None

    for i, tau in enumerate(np.linspace(tau_min, tau_max, num_tau)):
        typer.echo(f"\nClustering on {len(x)} data-points with threshold={tau}.")
        if method is Method.h0:
            merge_out = merge_h0(neighbor_graph=graph, density_map=density_map, threshold=tau)
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


if __name__ == "__main__":
    typer.run(main)
