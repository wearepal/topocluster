from __future__ import annotations
from abc import abstractstaticmethod
from typing import Any, Callable, ClassVar, Final, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from kit import implements
from kit.torch import prop_random_split
from topocluster.data.utils import (
    BinarizedLabelDataset,
    ImageDims,
    ImageDims,
    adaptive_collate,
)


__all__ = [
    "DataModule",
    "UMNISTDataModule",
    "VisionDataModule",
]


class DataModule(pl.LightningDataModule):

    train_data: Dataset
    val_data: Dataset
    test_data: Dataset
    dims: int | ImageDims

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        val_batch_size: int | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[List[Any]], Any] = adaptive_collate,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = test_batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    @abstractstaticmethod
    def num_classes() -> int:
        ...

    @abstractstaticmethod
    def num_subgroups() -> int:
        ...

    @implements(pl.LightningDataModule)
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )


class VisionDataModule(DataModule):

    dims: ImageDims

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        collate_fn: Callable[[List[Any]], Any] = adaptive_collate,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


class UMNISTDataModule(VisionDataModule):
    undersampling_props: Final[ClassVar[dict[str, float]]] = {"8": 0.05}
    threshold: Final[int] = 5

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        val_pcnt: float = 0.2,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )
        self.val_pcnt = val_pcnt
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_subgroups(self) -> int:
        return 5

    @implements(pl.LightningDataModule)
    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def _undersample(self, dataset: MNIST | Subset[MNIST]) -> Subset:
        if isinstance(dataset, Subset):
            targets = dataset.dataset.targets[dataset.indices]  # type: ignore
        else:
            targets = dataset.targets
        inds_to_keep = torch.ones(len(targets), dtype=torch.long)
        for class_, prop in self.undersampling_props.items():
            class_ = int(class_)  # hydra doesn't allow ints as keys, so we have to cast
            if not (0 <= prop <= 1):
                raise ValueError("Undersampling proportions must be between 0 and 1.")
            class_inds = (targets == class_).nonzero()
            n_matches = len(class_inds)
            num_to_drop = round(1 - prop * (n_matches - 1))
            to_drop = torch.randperm(n_matches) < num_to_drop  # type: ignore
            inds_to_keep[class_inds[to_drop]] = 0

        return Subset(dataset=dataset, indices=inds_to_keep.tolist())

    @implements(pl.LightningDataModule)
    def setup(self, stage: str | None = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.val_data, self.train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = BinarizedLabelDataset(
                self._undersample(self.train_data), threshold=self.threshold
            )
            self.val = BinarizedLabelDataset(self.val_data, threshold=self.threshold)
            self.dims = ImageDims(*self.train_data[0][0].shape)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = BinarizedLabelDataset(
                MNIST(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self.transform,
                ),
                threshold=self.threshold,
            )
            self.dims = ImageDims(*getattr(self, "dims", self.test_data[0][0].shape))
