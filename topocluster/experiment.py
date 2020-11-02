# """Main training file"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.models.autoencoder import AutoEncoder
import wandb

__all__ = ["Experiment"]


class Experiment(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        encoder: AutoEncoder,
        clusterer: Clusterer,
        epochs: int,
        pretrain_epochs: int,
        lr: float = 1.0e-3,
        use_wandb: bool = False,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer

        self.trainer = pl.Trainer(max_epochs=epochs)
        self.pt_trainer = pl.Trainer(max_epochs=pretrain_epochs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch[0] if isinstance(batch, Sequence) else batch
        encoding = self.encoder(x)
        _ = self.clusterer.fit_transform(encoding)
        return encoding.sum()

    def start(self, raw_config: Optional[Dict[str, Any]]):
        self.datamodule.setup()
        if self.hparams.use_wandb:
            wandb.init(entity="predictive-analytics-lab", project="topocluster", config=raw_config)
        pl.seed_everything(seed=self.hparams.seed)
        self.encoder.build(self.datamodule.dims)
        self.pt_trainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
