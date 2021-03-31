# """Main training file"""
from __future__ import annotations
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim import AdamW, Optimizer
from torch.tensor import Tensor
import wandb

from kit import implements
from topocluster.clustering.common import Clusterer
from topocluster.clustering.tomato import Tomato
from topocluster.data.datamodules import DataModule
from topocluster.data.utils import Batch
from topocluster.metrics import compute_metrics
from topocluster.models.base import Encoder


__all__ = ["Experiment"]


class Experiment(pl.LightningModule):

    artifacts_dir: ClassVar[Path] = Path("artifacts_dir")

    def __init__(
        self,
        datamodule: DataModule,
        encoder: Encoder,
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
        self.encoder_loss_weight = recon_loss_weight
        self.clust_loss_weight = clust_loss_weight

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        encoding = self.encoder(batch.x)
        loss_dict = {}
        if self.encoder_loss_weight > 0:
            loss_dict.update(self.encoder.get_loss(encoding, batch, prefix="train/"))
        if self.clust_loss_weight > 0:
            loss_dict.update(self.clusterer.get_loss(x=encoding, prefix="train/"))
        total_loss = cast(Tensor, sum(loss_dict.values()))
        loss_dict["train/total_loss"] = total_loss
        self.logger.experiment.log(loss_dict)

        return total_loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        encoding = self.encoder(batch.x)
        # Defer clustering until the end of the epoch so that clustering can take into account
        # all of the data during fitting
        return encoding, batch.s, batch.y

    @implements(pl.LightningModule)
    def validation_epoch_end(
        self,
        outputs: list[tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        self._val_test_epoch_end(outputs=outputs, stage="val")

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch, batch_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.validation_step(batch, batch_idx)

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[tuple[Tensor, Tensor, Tensor]]) -> None:
        self._val_test_epoch_end(outputs=outputs, stage="test")

    def _val_test_epoch_end(
        self, outputs: list[tuple[Tensor, Tensor, Tensor]], stage: Literal["val", "test"]
    ) -> None:
        encodings, subgroup_inf, targets = tuple(zip(*outputs))
        encodings, subgroup_inf, targets = (
            torch.cat(encodings, dim=0),
            torch.cat(subgroup_inf, dim=0),
            torch.cat(targets, dim=0),
        )

        self.print("Clustering using all data.")
        preds = self.clusterer(encodings)[0]
        logging_dict = compute_metrics(
            preds=preds, subgroup_inf=subgroup_inf, targets=targets, prefix=stage
        )

        if isinstance(self.clusterer, Tomato):
            pers_diagrams = {
                f"{stage}/pers_diagram_[thresh={self.clusterer.threshold}]": wandb.Image(
                    self.clusterer.plot()
                )
            }
            self.print("Computing the persistence diagram with threshold=1.0")
            self.clusterer(encodings, threshold=1.0)
            pers_diagrams[f"{stage}/pers_diagram_[thresh=1.0]"] = wandb.Image(self.clusterer.plot())
            self.logger.experiment.log(pers_diagrams)

        self.log_dict(logging_dict)

    def start(self, raw_config: dict[str, Any] | None = None):
        self.datamodule.setup()
        self.datamodule.prepare_data()
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

        logger = WandbLogger(
            entity="predictive-analytics-lab",
            project="topocluster",
            offline=self.log_offline,
        )
        if raw_config is not None:
            logger.log_hyperparams(raw_config)
        self.pretrainer.logger = logger
        self.trainer.logger = logger

        self.trainer.callbacks.append(
            ModelCheckpoint(
                monitor="val/Accuracy",
                dirpath=self.artifacts_dir,
                save_top_k=1,
                filename="best",
                mode="max",
            )
        )

        pl.seed_everything(seed=self.seed)
        self.encoder.build(self.datamodule)
        self.clusterer.build(encoder=self.encoder, datamodule=self.datamodule)
        # self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
        self.trainer.test(self, datamodule=self.datamodule)
