from dataclasses import dataclass
from typing import Any, ClassVar, Final, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets.omniglot import Omniglot
from torchvision.datasets.svhn import SVHN

from topocluster.data.utils import prop_random_split


__all__ = [
    "DataModule",
    "MNISTDataModule",
    "CIFAR10DataModule",
    "CIFAR100DataModule",
    "SVHNDataModule",
    "OmniglotDataModule",
]


MASK_VALUE: Final = -1


class MaskedLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, threshold: Optional[int] = None) -> None:
        self.dataset = dataset
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        x, y = self.dataset[index]
        if self.threshold is None or y >= self.threshold:
            y = MASK_VALUE
        return x, y


class DataModule(pl.LightningDataModule):

    train_data: Dataset
    val_data: Dataset
    test_data: Dataset
    dims: Tuple[int, int, int]
    num_classes: ClassVar[int]

    def __init__(
        self,
        label_threshold: int,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.label_threshold = label_threshold

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class MNISTDataModule(DataModule):

    num_classes: ClassVar[int] = 10

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        val_pcnt: float = 0.2,
        label_threshold: int = 5,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            label_threshold=label_threshold,
        )
        self.val_pcnt = val_pcnt
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.val_data, train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = MaskedLabelDataset(train_data, threshold=self.label_threshold)
            self.dims = self.train_data[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )
            self.dims = getattr(self, "dims", self.test_data[0][0].shape)


class CIFAR10DataModule(DataModule):

    num_classes: ClassVar[int] = 10

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        val_pcnt: float = 0.2,
        label_threshold: int = 5,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            label_threshold=label_threshold,
        )
        self.val_pcnt = val_pcnt
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_data(self) -> None:
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
            self.val_data, train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = MaskedLabelDataset(train_data, threshold=self.label_threshold)
            self.dims = self.train_data[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = CIFAR10(
                self.data_dir, train=False, download=True, transform=self.transform
            )


class CIFAR100DataModule(DataModule):

    num_classes: ClassVar[int] = 100

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        val_pcnt: float = 0.2,
        label_threshold: int = 80,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            label_threshold=label_threshold,
        )
        self.val_pcnt = val_pcnt
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_data(self) -> None:
        # download
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = CIFAR100(self.data_dir, train=True, download=True, transform=self.transform)
            self.val_data, train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = MaskedLabelDataset(train_data, threshold=self.label_threshold)
            self.dims = self.train_data[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = CIFAR100(
                self.data_dir, train=False, download=True, transform=self.transform
            )


class SVHNDataModule(DataModule):

    num_classes: ClassVar[int] = 10

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        val_pcnt: float = 0.2,
        label_threshold: int = 5,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            label_threshold=label_threshold,
        )
        self.val_pcnt = val_pcnt
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self) -> None:
        # download
        SVHN(root=self.data_dir, split="train", download=True)
        SVHN(root=self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            all_data = SVHN(self.data_dir, split="train", download=True, transform=self.transform)
            self.val_data, train_data = prop_random_split(all_data, props=(self.val_pcnt,))
            self.train_data = MaskedLabelDataset(train_data, threshold=self.label_threshold)
            self.dims = self.train_data[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_data = SVHN(
                self.data_dir, split="test", download=True, transform=self.transform
            )


class OmniglotDataModule(DataModule):

    num_classes: ClassVar[int] = 50

    def __init__(
        self,
        data_dir: str = "./",
        train_batch_size: int = 256,
        test_batch_size: int = 1000,
        num_workers: int = 0,
        label_threshold: int = 5,
    ):
        super().__init__(
            data_dir=data_dir,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            label_threshold=label_threshold,
        )
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self):
        # download
        Omniglot(root=self.data_dir, background=True, download=True)
        Omniglot(root=self.data_dir, background=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        # Assign Train/val split(s) for use in Dataloaders
        background = Omniglot(root=self.data_dir, background=True, download=True)
        evaluation = Omniglot(root=self.data_dir, background=False, download=True)
        all_data = ConcatDataset([background, MaskedLabelDataset(evaluation)])
        # TODO: Create validation and test sets
        self.train_data = all_data
        self.dims = self.train_data[0][0].shape
