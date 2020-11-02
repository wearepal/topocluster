# """Main training file"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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
        trainer: pl.Trainer,
        pt_trainer: pl.Trainer,
        lr: float = 1.0e-3,
        use_wandb: bool = False,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.save_hyperparameters("lr", "use_wandb", "seed")
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer
        self.trainer = trainer
        self.pt_trainer = pt_trainer

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.encoder.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch[0] if isinstance(batch, Sequence) else batch
        encoding = self.encoder(x)
        _ = self.clusterer.fit_transform(encoding)
        loss = encoding.sum()
        self.log("train_loss", loss)
        tqdm_dict = {"loss": loss}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def start(self, raw_config: Optional[Dict[str, Any]] = None):
        self.datamodule.setup()
        if self.hparams.use_wandb:
            wandb.init(entity="predictive-analytics-lab", project="topocluster", config=raw_config)
            self.pt_trainer.logger = WandbLogger()
            self.trainer.logger = WandbLogger()
        pl.seed_everything(seed=self.hparams.seed)
        self.encoder.build(self.datamodule.dims)
        self.pt_trainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self.encoder, datamodule=self.datamodule)
