"""Functions for computing metrics."""
from __future__ import annotations

import numpy as np
from torch import Tensor

import ethicml as em
import pandas as pd
from topocluster.clustering.utils import (
    compute_optimal_assignments,
    encode_arr_with_dict,
)

__all__ = ["compute_metrics"]


def compute_metrics(
    preds: Tensor, subgroup_inf: Tensor, targets: Tensor, prefix: str
) -> dict[str, float]:
    logging_dict: dict[str, float] = {}

    # Convert from torch to numpy
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.cpu().numpy()
    subgroup_inf_np = subgroup_inf.cpu().numpy()

    total_acc, cluster_map = compute_optimal_assignments(
        labels_true=targets_np, labels_pred=preds_np
    )
    logging_dict[f"{prefix}/total_acc"] = total_acc
    aligned_preds = encode_arr_with_dict(preds_np, cluster_map)

    # ==================================== EthicML metrics ====================================
    aligned_preds = em.Prediction(hard=pd.Series(aligned_preds))
    aligned_preds._info = {}
    sens = pd.DataFrame(subgroup_inf_np.astype(np.float32), columns=["subgroup"])
    labels = pd.DataFrame(targets_np.flatten().astype(np.float32), columns=["superclass"])
    actual = em.DataTuple(x=sens, s=sens, y=labels)
    metrics = em.run_metrics(
        aligned_preds,
        actual,
        metrics=[em.Accuracy(), em.TPR(), em.TNR(), em.RenyiCorrelation()],
        per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR(), em.TNR()],
        diffs_and_ratios=False,
    )
    # replace the slash -- it's causing problems -- and add the prefix
    metrics = {f"{prefix}/" + k.replace("/", "÷"): v for k, v in metrics.items()}
    logging_dict.update(metrics)

    return logging_dict
