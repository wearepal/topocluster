# """Main training file"""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.clustering.dac import PseudoLabelLoss
from topocluster.clustering.loss import l2_centroidal_distance
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
        pretrainer: pl.Trainer,
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
        self.pretrainer = pretrainer

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        labeled = y != -1

        encoding = self.encoder(x)
        self.clusterer.fit(encoding).labels

        # purity_loss = l2_centroidal_distance(
        #     encoding[labeled], self.clusterer.centroids, y[labeled]
        # )
        purity_loss = encoding.sum()

        self.log("purity_loss", purity_loss, on_step=True, prog_bar=True, logger=True)

        return purity_loss

    def start(self, raw_config: Optional[Dict[str, Any]] = None):
        self.datamodule.setup()
        if self.hparams.use_wandb:
            wandb.init(entity="predictive-analytics-lab", project="topocluster", config=raw_config)
            self.pretrainer.logger = WandbLogger()
            self.trainer.logger = WandbLogger()
        pl.seed_everything(seed=self.hparams.seed)
        self.encoder.build(self.datamodule.dims)
        self.clusterer.build(self.encoder.latent_dim, self.datamodule.num_classes)
        # self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
