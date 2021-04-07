# """Main training file"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, cast

from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim import AdamW, Optimizer
from torch.tensor import Tensor
from umap import UMAP
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
        reducer: Optional[UMAP],
        lr: float = 1.0e-3,
        weight_decay: float = 9,
        log_offline: bool = False,
        seed: Optional[int] = 42,
        enc_loss_w: float = 1.0,
        clust_loss_w: float = 1.0,
        exp_group: Optional[str] = None,
        train_eval_freq: int = 1,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.log_offline = log_offline
        self.exp_group = exp_group
        self.seed = seed
        self.train_eval_freq = train_eval_freq
        self.checkpoint_path = checkpoint_path
        # Components
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer
        self.reducer = reducer
        # Trainers
        self.trainer = trainer
        self.pretrainer = pretrainer
        # Optimizer configuration
        self.lr = lr
        self.weight_decay = weight_decay
        # Pre-factors
        self.enc_loss_w = enc_loss_w
        self.clust_loss_w = clust_loss_w

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_loss(
        self, encoding: Tensor, batch: Batch, stage: Literal["train", "val", "test"]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}
        total_loss = encoding.new_zeros(())
        if self.enc_loss_w > 0:
            enc_loss_dict = self.encoder.get_loss(encoding, batch, prefix=stage)
            loss_dict.update(enc_loss_dict)
            total_loss += self.enc_loss_w * sum(enc_loss_dict.values())
        if self.clust_loss_w > 0:
            clust_loss_dict = self.clusterer.get_loss(x=encoding, prefix=stage)
            loss_dict.update(clust_loss_dict)
            total_loss += self.enc_loss_w * sum(clust_loss_dict.values())
        loss_dict[f"{stage}/total_loss"] = total_loss
        return total_loss, loss_dict

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> dict[str, Tensor | None]:
        encoding = cast(Tensor, self.encoder(batch.x))
        res_dict = {
            "encoding": encoding,
            "subgroup_inf": batch.s,
            "superclass_inf": batch.y,
        }
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="train")
        self.log_dict(loss_dict)
        res_dict["loss"] = total_loss

        return res_dict

    @implements(pl.LightningModule)
    def training_epoch_end(
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        if (self.current_epoch % self.train_eval_freq) == 0:
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
        if self.reducer is not None:
            encodings = torch.as_tensor(
                self.reducer.fit_transform(encodings.detach().cpu().numpy()),
                device=encodings.device,
            )
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
            if self.clusterer.threshold < 1.0:
                self.clusterer(encodings, threshold=1.0)
                pers_diagrams[f"{stage}/pers_diagram_[thresh=1.0]"] = wandb.Image(
                    self.clusterer.plot()
                )
            plt.close("all")

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
            group=self.clusterer.__class__.__name__ if self.exp_group is None else self.exp_group,
        )
        hparams = {"artifacts_dir": self.artifacts_dir.resolve(), "cwd": os.getcwd()}
        if raw_config is not None:
            hparams.update(raw_config)
        logger.log_hyperparams(hparams)
        self.pretrainer.logger = logger
        self.trainer.logger = logger

        checkpointer = ModelCheckpoint(
            monitor="val/total_loss",
            dirpath=self.artifacts_dir,
            save_top_k=1,
            filename="best",
            mode="max",
        )
        self.pretrainer.callbacks.append(checkpointer)
        self.trainer.callbacks.append(checkpointer)

        # PRNG seeding
        pl.seed_everything(seed=self.seed)
        # Build the encoder
        self.encoder.build(self.datamodule)
        # Build the clusterer
        self.clusterer.build(encoder=self.encoder, datamodule=self.datamodule)
        # Load weights/hparams from checkpoint-path if provided
        if self.checkpoint_path is not None:
            self = self.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                datamodule=self.datamodule,
                encoder=self.encoder,
                clusterer=self.clusterer,
                pretrainer=self.pretrainer,
                trainer=self.trainer,
            )
        # Pre-training phase
        self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        # Training phase
        self.trainer.fit(self, datamodule=self.datamodule)
        # Testing phase
        self.trainer.test(self, datamodule=self.datamodule)
        # Manually call exit for multirun compatibility
        logger.experiment.__exit__(None, 0, 0)
