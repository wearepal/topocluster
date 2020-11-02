from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


__all__ = ["MNISTDataModule"]


class MNISTDataModule(pl.LightningDataModule):
    dims: Tuple[int, int, int]

    def __init__(
        self,
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

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            self.dims = self.mnist_train[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )
            self.dims = getattr(self, "dims", self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
