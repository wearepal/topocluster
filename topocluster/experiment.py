# """Main training file"""
from __future__ import annotations
from typing import Any, Optional, cast

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.optim import AdamW, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.clustering.utils import compute_optimal_assignments
from topocluster.data.datamodules import DataModule
from topocluster.data.utils import Batch
from topocluster.models.autoencoder import AutoEncoder
from topocluster.utils.interface import implements


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
        weight_decay: float = 9,
        log_offline: bool = False,
        seed: Optional[int] = 42,
        recon_loss_weight: float = 1.0,
        clust_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.log_offline = log_offline
        self.seed = seed
        # Components
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer
        # Trainers
        self.trainer = trainer
        self.pretrainer = pretrainer
        # Optimizer configuration
        self.lr = lr
        self.weight_decay = weight_decay
        # Pre-factors
        self.recon_loss_weight = recon_loss_weight
        self.clust_loss_weight = clust_loss_weight

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x, y = batch
        encoding = self.encoder(x)
        hard_labels, soft_labels = self.clusterer(encoding)
        loss_dict = {}
        if self.recon_loss_weight > 0:
            loss_dict.update(self.encoder.get_loss(encoding, x, prefix="train"))
        if self.clust_loss_weight > 0:
            loss_dict.update(
                self.clusterer.get_loss(
                    x=encoding,
                    soft_labels=soft_labels,
                    hard_labels=hard_labels,
                    y=y,
                    prefix="train",
                )
            )
        total_loss = cast(Tensor, sum(loss_dict.values()))
        loss_dict["train/total_loss"] = total_loss
        self.logger.experiment.log(loss_dict)

        return total_loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, float]:
        x, y = batch
        y_np = y.cpu().numpy()

        encoding = self.encoder(x)
        preds = self.clusterer(encoding)[0].cpu().detach().numpy()

        metrics = {
            "val/ARI": adjusted_rand_score(labels_true=y_np, labels_pred=preds),
            "val/NMI": normalized_mutual_info_score(labels_true=y_np, labels_pred=preds),  # type: ignore
            "val/Accuracy": compute_optimal_assignments(
                labels_true=y_np, labels_pred=preds, num_classes=self.datamodule.num_classes
            )[0],
        }
        self.log_dict(metrics, prog_bar=True, logger=False)
        self.logger.experiment.log(metrics)

        return metrics

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch, batch_idx: int) -> dict[str, float]:
        val_metrics = self.validation_step(batch, batch_idx)
        test_metrics = {
            "test/ARI": val_metrics["val/ARI"],
            "test/NMI": val_metrics["val/NMI"],
            "test/Accuracy": val_metrics["val/Accuracy"],
        }
        self.logger.experiment.log(test_metrics)

        return test_metrics

    def start(self, raw_config: dict[str, Any] | None = None):
        self.datamodule.setup()
        logger = WandbLogger(
            entity="predictive-analytics-lab",
            project="topocluster",
            offline=self.log_offline,
        )
        if raw_config is not None:
            logger.log_hyperparams(raw_config)
        self.pretrainer.logger = logger
        self.trainer.logger = logger

        pl.seed_everything(seed=self.seed)
        self.encoder.build(self.datamodule.dims)
        self.clusterer.build(self.encoder.latent_dim, self.datamodule.num_classes)

        self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
        self.trainer.test(self, datamodule=self.datamodule)
