from __future__ import annotations
from abc import abstractmethod, abstractstaticmethod
from typing import Any, ClassVar, List, cast

import pytorch_lightning as pl
import torch
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision.datasets import MNIST

from kit import implements
from kit.torch import prop_random_split
from topocluster.data.utils import (
    Batch,
    BinarizedLabelDataset,
    DataTransformer,
    ImageDims,
    adaptive_collate,
    cast_collation,
)


__all__ = [
    "DataModule",
    "MNISTDataModule",
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
        train_batch_sampler: Sampler[List[int]] | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = test_batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        self._collate_fn = cast_collation(adaptive_collate, Batch)
        self.train_batch_sampler = train_batch_sampler

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abstractstaticmethod
    def num_subgroups(self) -> int:
        ...

    @implements(pl.LightningDataModule)
    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        dl_kwargs: dict[str, Any] = {
            "shuffle": shuffle and self.train_batch_sampler is None,
            "drop_last": self.train_batch_sampler is None,
        }
        if self.train_batch_sampler is None:
            dl_kwargs["batch_size"] = self.train_batch_size
        return DataLoader(
            self.train_data,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            batch_sampler=self.train_batch_sampler,
            **dl_kwargs,
        )

    @implements(pl.LightningDataModule)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    @implements(pl.LightningDataModule)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self._collate_fn,
        )


class VisionDataModule(DataModule):

    dims: ImageDims

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )


class MNISTDataModule(VisionDataModule):
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

    @staticmethod
    def _transform() -> transforms.Compose:
        transform_ls = [transforms.Resize((32, 32)), transforms.Normalize((0.1307,), (0.3081,))]
        # test_transform_list.insert(0, ))
        return transforms.Compose(transform_ls)

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

    @implements(pl.LightningDataModule)
    def setup(self, stage: str | None = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = BinarizedLabelDataset(
                MNIST(self.data_dir, train=True, download=True, transform=self._transform()),
                threshold=5,
            )
            self.val_data, self.train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            sample = cast(Tensor, self.train_data[0][0])
            self.dims = ImageDims(*sample.shape)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = BinarizedLabelDataset(
                MNIST(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self._transform(),
                ),
                threshold=5,
            )
            self.dims = ImageDims(*getattr(self, "dims", self.test_data[0][0].shape))


class UMNISTDataModule(VisionDataModule):
    undersampling_props: ClassVar[dict[str, float]] = {"8": 0.05}
    threshold: ClassVar[int] = 5

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

    @staticmethod
    def _transform(train: bool) -> transforms.Compose:
        test_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        test_transform_list.insert(0, transforms.Resize((32, 32)))
        if not train:
            return transforms.Compose(test_transform_list)

        train_transform_list = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + test_transform_list
        return transforms.Compose(train_transform_list)

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
        inds_to_keep_mask = torch.ones(len(targets), dtype=torch.long)
        for class_, prop in self.undersampling_props.items():
            class_ = int(class_)  # hydra doesn't allow ints as keys, so we have to cast
            if not (0 <= prop <= 1):
                raise ValueError("Undersampling proportions must be between 0 and 1.")
            class_inds = (targets == class_).nonzero()
            n_matches = len(class_inds)
            num_to_drop = round((1 - prop) * (n_matches - 1))
            to_drop = torch.randperm(n_matches) < num_to_drop  # type: ignore
            inds_to_keep_mask[class_inds[to_drop]] = 0

        inds_to_keep = inds_to_keep_mask.nonzero().squeeze(-1).tolist()

        return Subset(dataset=dataset, indices=inds_to_keep)

    @implements(pl.LightningDataModule)
    def setup(self, stage: str | None = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = MNIST(self.data_dir, train=True, download=True)
            val_data, train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = DataTransformer(
                BinarizedLabelDataset(self._undersample(train_data), threshold=self.threshold),  # type: ignore
                self._transform(True),
            )
            self.val_data = DataTransformer(
                BinarizedLabelDataset(val_data, threshold=self.threshold),
                self._transform(False),
            )
            sample = cast(Tensor, self.train_data[0][0])
            self.dims = ImageDims(*sample.shape)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = BinarizedLabelDataset(
                MNIST(
                    self.data_dir,
                    train=False,
                    download=True,
                    transform=self._transform(False),
                ),
                threshold=self.threshold,
            )
            self.dims = ImageDims(*getattr(self, "dims", self.test_data[0][0].shape))
