from typing import Any, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, Subset, random_split
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

__all__ = ["RandomSampler", "adaptive_collate", "filter_by_labels", "prop_random_split"]


def prop_random_split(dataset: Dataset, props: Sequence[float]) -> List[Subset]:
    len_ = len(dataset)
    if (sum_ := (np.sum(props)) > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    return random_split(dataset, section_sizes)


def filter_by_labels(
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], labels: Set[int]
) -> Subset:
    """Filter samples from a dataset by labels."""
    indices: List[int] = []
    for _, _, y in dataset:
        if (label := int(y.numpy())) in labels:
            indices.append(label)
    return Subset(dataset, indices)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())

        return iter(torch.randperm(n)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def adaptive_collate(batch: Any) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if (ndims := elem.dim()) > 0 and ndims % 2 == 0:
            return torch.cat(batch, dim=0, out=out)
        else:
            return torch.stack(batch, dim=0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return adaptive_collate([torch.as_tensor(b) for b in batch])
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(adaptive_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return [adaptive_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))
