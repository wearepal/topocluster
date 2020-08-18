from typing import Tuple, Union, List, Optional, Mapping, Dict
import torch.nn as nn

from topocluster import layers
from topocluster.configs import ClusterArgs
from topocluster.models import Classifier
from topocluster.models.configs import ModelFn

__all__ = ["build_classifier"]


def build_classifier(
    input_shape: Tuple[int, ...],
    target_dim: int,
    model_fn: ModelFn,
    model_kwargs: Mapping[str, Union[float, str, bool]],
    optimizer_kwargs: Optional[Dict[str, float]] = None,
    num_heads: int = 1,
) -> Union[nn.ModuleList, Classifier]:
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    if num_heads > 1:
        heads: List[Classifier] = []
        for _ in range(num_heads):
            heads.append(
                Classifier(
                    model_fn(in_dim, target_dim, **model_kwargs),
                    num_classes=num_classes,
                    optimizer_kwargs=optimizer_kwargs,
                )
            )
        classifier = nn.ModuleList(heads)
    else:
        classifier = Classifier(
            model_fn(in_dim, target_dim, **model_kwargs),
            num_classes=num_classes,
            optimizer_kwargs=optimizer_kwargs,
        )

    return classifier
