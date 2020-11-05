# """Main training file"""
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.metrics.functional as FM
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.clustering.utils import count_occurances, find_optimal_assignments
from topocluster.data.data_modules import DataModule
from topocluster.models.autoencoder import AutoEncoder
import wandb


__all__ = ["Experiment"]


class Experiment(pl.LightningModule):
    def __init__(
        self,
        datamodule: DataModule,
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

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, y = batch
        y_np = y.cpu().numpy()

        encoding = self.encoder(x).cpu()
        preds = self.clusterer(encoding).cpu().detach().numpy()

        metrics = {}
        metrics["ARI"] = adjusted_rand_score(labels_true=y_np, labels_pred=preds)
        metrics["NMI"] = normalized_mutual_info_score(labels_true=y_np, labels_pred=preds)
        self.log_dict(metrics)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        labeled = y != -1

        encoding = self.encoder(x)
        recon_loss = self.encoder.get_loss(encoding, x)
        cluster_loss = self.clusterer.get_loss(encoding)

        purity_loss = F.cross_entropy(self.clusterer.soft_labels[labeled], y[labeled])

        self.log_dict(
            {"recon_loss": recon_loss, "cluster_loss": cluster_loss, "purity_loss": purity_loss}
        )

        loss = recon_loss + purity_loss
        if cluster_loss is not None:
            loss += cluster_loss

        return loss

    def start(self, raw_config: Optional[Dict[str, Any]] = None):
        self.datamodule.setup()
        if self.hparams.use_wandb:
            wandb.init(entity="predictive-analytics-lab", project="topocluster", config=raw_config)
            self.pretrainer.logger = WandbLogger()
            self.trainer.logger = WandbLogger()
        pl.seed_everything(seed=self.hparams.seed)

        self.encoder.build(self.datamodule.dims)
        self.clusterer.build(self.encoder.latent_dim, self.datamodule.num_classes)

        self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
