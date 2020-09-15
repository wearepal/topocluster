from typing import Any, Dict

import numpy as np
import warnings
from torch.distributed.rpc.backend_registry import init_backend
from torch.tensor import Tensor
import umap
import wandb
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from topocluster.configs import ClusterArgs
from topocluster.models import Encoder
from topocluster.optimisation.utils import count_occurances, find_assignment, get_class_id
from topocluster.optimisation.evaluation import encode_dataset
from topocluster.optimisation.unsupervised.topograd import TopoCluster
from topocluster.utils import ClusterResults, wandb_log

from .k_means import run_kmeans_faiss

__all__ = ["cluster"]


def cluster(
    args: ClusterArgs, encoder: Encoder, context_data: Dataset, num_clusters: int, s_count: int
) -> ClusterResults:
    # encode the training set with the encoder
    encoded = encode_dataset(args, context_data, encoder)
    # create data loader with one giant batch
    data_loader = DataLoader(encoded, batch_size=len(encoded), shuffle=False)
    encoded, s, y = next(iter(data_loader))

    logging_dict: Dict[str, Any] = {}

    if args.method == "kmeans":
        preds = run_kmeans_faiss(
            encoded,
            nmb_clusters=num_clusters,
            cuda=str(args._device) != "cpu",
            n_iter=args.epochs,
            verbose=True,
        )
    else:
        clusterer = TopoCluster(
            k_kde=args.tc_k_kde,
            k_vrc=args.tc_k_vrc,
            scale=args.tc_scale,
            batch_size=args.tc_batch_size,
            umap_kwargs=args.tc_umap_kwargs,
        )
        preds, barcode = clusterer.fit(encoded, threshold=args.tc_threshold)
        pd = clusterer.plot_pd(barcode, dpi=100)
        logging_dict["persistence_diagram"] = wandb.Image(pd)
        plt.close(pd)

    if args.visualize_clusters:
        if isinstance(encoded, Tensor):
            encoded = encoded.cpu().detach().numpy()
        n_neighbours = args.tc_umap_kwargs["n_components"] if args.tc_umap_kwargs else None
        reducer = umap.UMAP(n_neighbors=n_neighbours, n_components=2, random_state=args.seed)
        reduced = reducer.fit_transform(encoded)
        cluster_viz, ax = plt.subplots(dpi=100)
        ax.scatter(reduced[:, 0], reduced[:, 1], c=preds.cpu().detach().numpy(), cmap="tab10")
        ax.set_title(f"UMAP-reduced Clusters, threshold={args.tc_threshold}")
        logging_dict["cluster_viz"] = wandb.Image(cluster_viz)
        plt.close(cluster_viz)
    try:
        counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
        counts, _ = count_occurances(counts, preds.cpu().numpy(), s, y, s_count, args.cluster)
        context_acc, _, logging_dict_t = find_assignment(counts, preds.size(0))
        logging_dict.update(logging_dict_t)
    except IndexError:
        warnings.warn(
            "Clustering purity could not be computed because the number of predicted "
            "clusters exceeds the number of groun-truth clusters."
        )
    prepared = (
        f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in logging_dict.items()
    )
    print(" | ".join(prepared))

    wandb_log(args, logging_dict)

    return ClusterResults(
        flags=args.as_dict(),
        cluster_ids=preds,
        class_ids=get_class_id(s=s, y=y, s_count=s_count, to_cluster=args.cluster),
        context_acc=context_acc,
    )
