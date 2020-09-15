from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import umap
import wandb
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset

from topocluster.configs import ClusterArgs
from topocluster.models import Encoder
from topocluster.optimisation.evaluation import encode_dataset
from topocluster.optimisation.unsupervised.topograd import TopoCluster
from topocluster.optimisation.utils import (
    ClusterResults,
    count_occurances,
    find_assignment,
    get_class_id,
)
from topocluster.utils import wandb_log

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
    ground_truth = get_class_id(s=s, y=y, s_count=s_count, to_cluster="both").detach().cpu().numpy()
    n_samples = ground_truth.shape[0]
    logging_dict: Dict[str, Any] = {}

    def _compute_purity(
        _preds: Union[Tensor, np.ndarray[np.float32]], _suffix: str = ""
    ) -> Tuple[float, float, float]:
        if isinstance(_preds, Tensor):
            _preds = _preds.detach().cpu().numpy()
        #  The Hungarian algorithm is only valid when there is a 1:1 correspondence between
        # clusters and the ground-truth classes
        _logging_dict_t: Dict[str, Union[float, str]] = {}
        try:
            _counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
            _counts, _ = count_occurances(_counts, preds.cpu().numpy(), s, y, s_count, args.cluster)
            _acc, _, _logging_dict_t = find_assignment(_counts, preds.size(0))
        except IndexError:
            _acc = float("nan")

        _ari = adjusted_rand_score(labels_true=ground_truth, labels_pred=_preds)
        _nmi = normalized_mutual_info_score(labels_true=ground_truth, labels_pred=_preds)
        _logging_dict_t["ARI"] = _ari
        _logging_dict_t["NMI"] = _nmi

        if _suffix:
            _logging_dict_t = {f"{key}_{_suffix}": value for key, value in _logging_dict_t.items()}
            logging_dict.update(_logging_dict_t)
        return _acc, _ari, _nmi

    if args.visualize_clusters:
        n_neighbours = (
            min(args.tc_umap_kwargs["n_components"], n_samples - 1) if args.tc_umap_kwargs else None
        )
        reducer = umap.UMAP(n_neighbors=n_neighbours, n_components=2, random_state=args.seed)
        reduced = reducer.fit_transform(encoded)

        def _plot_clusters(_preds: np.ndarray[np.float32], _suffix: str = "") -> None:
            cluster_viz, ax = plt.subplots(dpi=100)
            ax.scatter(
                reduced[:, 0], reduced[:, 1], c=_preds, cmap="tab10"  # type: ignore[arg-type]
            )
            ax.set_title(f"UMAP-reduced Clusters, {_suffix}")
            key = "cluster_viz"
            if _suffix:
                key += f"_{_suffix}"
            logging_dict[key] = wandb.Image(cluster_viz)
            plt.close(cluster_viz)

        _plot_clusters(_preds=ground_truth, _suffix="[ground_truth]")

    if args.method == "kmeans":
        preds = run_kmeans_faiss(
            encoded,
            nmb_clusters=num_clusters,
            cuda=str(args._device) != "cpu",
            n_iter=args.epochs,
            verbose=True,
        )
        preds_np = preds.detach().cpu().numpy()
        if args.visualize_clusters:
            _plot_clusters(_preds=preds_np, _suffix="")  #  type: ignore

        acc, ari, nmi = _compute_purity(_preds=preds, _suffix="")
        results = ClusterResults(
            args=args.as_dict(),
            cluster_ids=Tensor(),
            class_ids=get_class_id(s=s, y=y, s_count=s_count, to_cluster=args.cluster),
            context_nmi=nmi,
            context_ari=ari,
            context_acc=acc,
        )
    else:
        clusterer = TopoCluster(
            k_kde=args.tc_k_kde,
            k_vrc=args.tc_k_vrc,
            scale=args.tc_scale,
            batch_size=args.tc_batch_size,
            umap_kwargs=args.tc_umap_kwargs,
        )
        thresholds = args.tc_thresholds or [1]
        assert thresholds

        all_acc: List[float] = []
        all_ari: List[float] = []
        all_nmi: List[float] = []

        results = ClusterResults(
            args=args.as_dict(),
            cluster_ids=Tensor(),
            class_ids=get_class_id(s=s, y=y, s_count=s_count, to_cluster=args.cluster),
        )

        best_score = float("-inf")
        for thresh in thresholds:
            suffix = f"[thresholds={thresh}"
            preds, barcode = clusterer.fit(encoded, threshold=thresh)
            pd = clusterer.plot_pd(barcode, dpi=100)
            logging_dict[f"persistence_diagram_{suffix}"] = wandb.Image(pd)
            plt.close(pd)

            preds_np = preds.detach().cpu().numpy()
            if args.visualize_clusters:
                _plot_clusters(_preds=preds_np, _suffix=suffix)

            acc, ari, nmi = _compute_purity(_preds=preds_np, _suffix=suffix)
            if acc != float("nan"):
                all_acc.append(acc)
            all_ari.append(ari)
            all_nmi.append(nmi)

            #  Use the predictions which yield the best score, defined as the average of the
            #  ARI and NMI (both maximal at 1)
            if (score := (0.5 * (ari + nmi))) > best_score:
                best_score = score
                results.cluster_ids = preds
                results.context_ari = ari
                results.context_nmi = nmi
                results.context_acc = acc

        if len(thresholds) > 1:
            fig, ax = plt.subplots(dpi=100)
            if all_acc:
                ax.plot(thresholds, all_acc, label="Acc.")
            ax.plot(thresholds, all_ari, label="ARI")
            ax.plot(thresholds, all_nmi, label="NMI")
            ax.set_xlabel("threshold")
            ax.legend()
            ax.set_title("Topocluster Threshold Sweep")
            logging_dict["threshold_sweep"] = wandb.Image(fig)
            plt.close(fig)

    prepared = (
        f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}" for k, v in logging_dict.items()
    )
    print(" | ".join(prepared))

    wandb_log(args, logging_dict)

    return results
