from __future__ import annotations
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Final, Optional, Protocol, Union

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

__all__ = [
    "IGNORE_INDEX",
    "ImageDims",
    "MaskedLabelDataset",
    "Transform",
    "DataTransformer",
    "SizedDatasetProt",
    "adaptive_collate",
    "cast_collation",
]


ImageDims = namedtuple("ImageDims", ["C", "H", "W"])
Batch = namedtuple("Batch", ["x", "s", "y"])
Transform = Callable[[Union[Image.Image, Tensor]], Tensor]
IGNORE_INDEX: Final = -100


def _cast(collate_fn: Transform, cast_to: type, inputs: list[Any]):
    return cast_to(*collate_fn(inputs))


def cast_collation(collate_fn: Transform, cast_to: type):
    return partial(_cast, collate_fn, cast_to)


class SizedDatasetProt(Protocol):
    """Typing Protocol for a SizedDataset (a Dataset defining a __len__ method)."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        ...


class DataTransformer(Dataset):
    def __init__(self, base_dataset: SizedDatasetProt, transforms: Transform):
        self.base_dataset = base_dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, ...]:
        data = self.base_dataset[index]
        if self.transforms is not None:
            data = (self.transforms(data[0]),) + data[1:]
        return tuple(data)


class MaskedLabelDataset(Dataset):
    def __init__(self, dataset: SizedDatasetProt, threshold: Optional[int] = None) -> None:
        self.dataset = dataset
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        x, y = self.dataset[index]
        if self.threshold is None or y >= self.threshold:
            y = IGNORE_INDEX
        return x, y


class BinarizedLabelDataset(Dataset):
    def __init__(self, dataset: SizedDatasetProt, threshold: int) -> None:
        self.dataset = dataset
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        x, y = self.dataset[index]
        y_bin = int(y >= self.threshold)
        return x, y, y_bin


def adaptive_collate(batch: list[Any]) -> Any:
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
