from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, Literal, Tuple, TypeVar, Union

from lapjv import lapjv
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader


__all__ = [
    "ClusterResults",
    "count_occurances",
    "count_parameters",
    "find_assignment",
    "get_class_id",
    "get_data_dim",
    "inf_generator",
    "restore_model",
    "save_model",
]


@dataclass
class ClusterResults:

    args: Dict[str, Any]
    cluster_ids: torch.Tensor
    class_ids: torch.Tensor
    context_ari: float = float("nan")
    context_nmi: float = float("nan")
    context_acc: float = float("nan")
    test_acc: float = float("nan")

    def save(self, save_path: Path) -> Path:
        save_dict = asdict(self)
        torch.save(save_dict, save_path)
        print(
            f"To make use of the generated cluster labels:\n--cluster-label-file {save_path.resolve()}"
        )
        return save_path


def save_model(
    save_dir: Path,
    model: nn.Module,
    epoch: int,
    sha: str,
    best: bool = False,
) -> Path:
    if best:
        filename = save_dir / "checkpt_best.pth"
    else:
        filename = save_dir / f"checkpt_epoch{epoch}.pth"
    save_dict = {
        "sha": sha,
        "model": model.state_dict(),
        "epoch": epoch,
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(filename: Path, model: Model) -> Tuple[Model, int]:
    chkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    args_chkpt = chkpt["args"]
    model.load_state_dict(chkpt["model"])
    return model, chkpt["epoch"]


def count_occurances(
    counts: np.ndarray,
    preds: np.ndarray,
    s: Tensor,
    y: Tensor,
    s_count: int,
    to_cluster: Literal["s", "y", "both"],
) -> Tuple[np.ndarray, Tensor]:
    """Count how often cluster IDs coincide with the class IDs.

    All possible combinations are accounted for.
    """
    class_id = get_class_id(s=s, y=y, s_count=s_count, to_cluster=to_cluster)
    indices, batch_counts = np.unique(
        np.stack([class_id.numpy().astype(np.int64), preds]), axis=1, return_counts=True
    )
    counts[tuple(indices)] += batch_counts
    return counts, class_id


def find_assignment(
    counts: np.ndarray, num_total: int
) -> Tuple[float, np.ndarray[np.int64], Dict[str, Union[float, str]]]:
    """Find an assignment of cluster to class such that the overall accuracy is maximized."""
    # row_ind maps from class ID to cluster ID: cluster_id = row_ind[class_id]
    # col_ind maps from cluster ID to class ID: class_id = row_ind[cluster_id]
    row_ind, col_ind, result = lapjv(-counts)
    best_acc = -result[0] / num_total
    assignment = (f"{class_id}->{cluster_id}" for class_id, cluster_id in enumerate(row_ind))
    logging_dict = {
        "Best acc": best_acc,
        "class ID -> cluster ID": ", ".join(assignment),
    }
    return best_acc, col_ind, logging_dict


def get_class_id(
    *, s: Tensor, y: Tensor, s_count: int, to_cluster: Literal["s", "y", "both"]
) -> Tensor:
    if to_cluster == "s":
        class_id = s
    elif to_cluster == "y":
        class_id = y
    else:
        class_id = y * s_count + s
    return class_id.squeeze()


def get_data_dim(data_loader: DataLoader) -> Tuple[int, ...]:
    x = next(iter(data_loader))[0]
    x_dim = x.shape[1:]

    return tuple(x_dim)


T = TypeVar("T")


def inf_generator(iterable: Iterable[T]) -> Iterator[T]:
    """Get DataLoaders in a single infinite loop.

    for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def count_parameters(model: nn.Module) -> int:
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
