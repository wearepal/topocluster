# """Main training file"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from omegaconf.omegaconf import MISSING
import pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.models.autoencoder import AutoEncoder
import wandb

__all__ = ["main"]


class Experiment(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        encoder: AutoEncoder,
        clusterer: Clusterer,
        epochs: int,
        pretrain_epochs: int,
        lr: float = 1.0e-3,
        use_wandb: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.use_wandb = use_wandb
        self.seed = seed

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch[0] if isinstance(batch, Sequence) else batch
        encoding = self.encoder(x)
        _ = self.clusterer.fit_transform(encoding)
        return encoding.sum()

    def start(self):
        self.datamodule.setup()
        # wandb.init(entity="predictive-analytics-lab", project="topocluster", config=None)
        pl.seed_everything(seed=self.seed)
        self.encoder.build(self.datamodule.dims)
        trainer_pt = pl.Trainer(max_epochs=self.pretrain_epochs)
        trainer_pt.fit(self.encoder, datamodule=self.datamodule)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self, datamodule=self.datamodule)
