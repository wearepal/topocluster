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

    def _get_loss(
        self, encoding: Tensor, batch: Batch, stage: Literal["train", "val", "test"]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total_loss = encoding.new_zeros(())
        loss_dict = {}
        if self.encoder_loss_weight > 0:
            enc_loss_dict = self.encoder.get_loss(encoding, batch, prefix=stage)
            loss_dict.update(enc_loss_dict)
            total_loss += self.encoder_loss_weight * sum(enc_loss_dict.values())
        if self.clust_loss_weight > 0:
            clust_loss_dict = self.clusterer.get_loss(x=encoding, prefix=stage)
            loss_dict.update(clust_loss_dict)
            total_loss += self.encoder_loss_weight * sum(clust_loss_dict.values())
        total_loss = cast(Tensor, sum(loss_dict.values()))
        loss_dict[f"{stage}/total_loss"] = total_loss
        return total_loss, loss_dict

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> dict[str, Tensor]:
        encoding = self.encoder(batch.x)
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="train")
        self.log_dict(loss_dict)

        return {
            "loss": total_loss,
            "encoding": encoding,
            "subgroup_inf": batch.s,
            "superclass_inf": batch.y,
        }

    @implements(pl.LightningModule)
    def training_epoch_end(
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        self._epoch_end(outputs=outputs, stage="train")

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        encoding = self.encoder(batch.x)
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="val")
        self.log_dict(loss_dict, on_epoch=True)
        return total_loss

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:
        encoding = self.encoder(batch.x)
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="test")
        self.log_dict(loss_dict, on_epoch=True)
        return total_loss

    def _epoch_end(
        self, outputs: list[dict[str, Tensor]], stage: Literal["train", "val", "test"]
    ) -> None:
        encodings, subgroup_inf, superclass_inf = [], [], []
        for step_outputs in outputs:
            encodings.append(step_outputs["encoding"])
            subgroup_inf.append(step_outputs["subgroup_inf"])
            superclass_inf.append(step_outputs["superclass_inf"])
        encodings, subgroup_inf, superclass_inf = (
            torch.cat(encodings, dim=0),
            torch.cat(subgroup_inf, dim=0),
            torch.cat(superclass_inf, dim=0),
        )

        self.print("Clustering using all data.")
        preds = self.clusterer(encodings)[0]
        logging_dict = compute_metrics(
            preds=preds,
            subgroup_inf=subgroup_inf,
            superclass_inf=superclass_inf,
            prefix=stage,
            num_subgroups=self.datamodule.num_subgroups,
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
                monitor="val/total_loss",
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
