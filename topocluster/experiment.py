# """Main training file"""
from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, cast

from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.tensor import Tensor
from torch.utils.data.dataloader import DataLoader
import wandb

from kit import implements
from topocluster.clustering.common import Clusterer
from topocluster.clustering.tomato import Tomato
from topocluster.data.datamodules import DataModule
from topocluster.data.sampling import GreedyCoreSetSampler
from topocluster.data.utils import Batch
from topocluster.metrics import compute_metrics
from topocluster.models.base import Encoder
from topocluster.reduction import RandomProjector, Reducer
from topocluster.utils.logging import EncodingProgbar


__all__ = ["Experiment"]


class Experiment(pl.LightningModule):

    artifacts_dir: ClassVar[Path] = Path("artifacts")

    def __init__(
        self,
        datamodule: DataModule,
        encoder: Encoder,
        clusterer: Clusterer,
        trainer: pl.Trainer,
        pretrainer: pl.Trainer,
        reducer: Reducer,
        sampler: GreedyCoreSetSampler,
        lr: float = 1.0e-3,
        weight_decay: float = 0,
        log_offline: bool = False,
        seed: Optional[int] = 42,
        enc_loss_w: float = 1.0,
        clust_loss_w: float = 1.0,
        exp_group: Optional[str] = None,
        train_eval_freq: int = 1000,
        enc_freeze_depth: Optional[int] = 0,
    ):
        super().__init__()
        self.log_offline = log_offline
        self.exp_group = exp_group
        self.seed = seed
        self.train_eval_freq = train_eval_freq
        # Components
        self.datamodule = datamodule
        self.encoder = encoder
        self.clusterer = clusterer
        self.reducer = reducer
        # Trainers
        self.trainer = trainer
        self._encoder_runner = copy.deepcopy(self.trainer)
        self._encoder_runner.callbacks.append(EncodingProgbar(trainer=self._encoder_runner))
        self.pretrainer = pretrainer
        self.train_step = 0
        # Optimizer configuration
        self.lr = lr
        self.weight_decay = weight_decay
        # Pre-factors
        self.enc_loss_w = enc_loss_w
        self.clust_loss_w = clust_loss_w
        self.enc_freeze_depth = enc_freeze_depth
        self.sampler = sampler

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_loss(
        self, encoding: Tensor, batch: Batch, stage: Literal["train", "val", "test"]
    ) -> tuple[Tensor | None, dict[str, Tensor]]:
        loss_dict: dict[str, Tensor] = {}
        total_loss: Tensor | None = encoding.new_zeros(())
        enc_loss_dict = self.encoder.get_loss(encoding, batch, prefix=stage)
        loss_dict.update(enc_loss_dict)
        if self.enc_loss_w > 0:
            total_loss += self.enc_loss_w * sum(enc_loss_dict.values())
        if self.clust_loss_w > 0:
            # Random projection can be differentiated through so we make exception to it
            # during end-to-end training
            if isinstance(self.reducer, RandomProjector):
                encoding = self.reducer.fit_transform(X=encoding)
            clust_loss_dict = self.clusterer.get_loss(x=encoding, prefix=stage)
            loss_dict.update(clust_loss_dict)
            total_loss += self.clust_loss_w * sum(clust_loss_dict.values())
        if not total_loss.requires_grad:
            total_loss = None
        else:
            loss_dict[f"{stage}/total_loss"] = total_loss
        return total_loss, loss_dict

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor | None:
        encoding = cast(Tensor, self.encoder(batch.x))
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="train")
        self.log_dict(loss_dict)
        self.train_step += 1
        return total_loss

    @implements(pl.LightningModule)
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        eff_train_step = self.train_step + 1
        if eff_train_step > 1 and (not (eff_train_step % self.train_eval_freq)):
            self._evaluate(stage="train")

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor | None:
        encoding = self.encoder(batch.x)
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="val")
        self.log_dict(loss_dict, on_epoch=True)
        return total_loss

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor | None:
        encoding = self.encoder(batch.x)
        total_loss, loss_dict = self._get_loss(encoding=encoding, batch=batch, stage="test")
        self.log_dict(loss_dict, on_epoch=True)
        return total_loss

    def _evaluate(self, stage: Literal["train", "val", "test"]) -> None:
        dl_kwargs = {"shuffle": False} if stage == "train" else {}
        train_batch_sampler = self.datamodule.train_batch_sampler
        self.datamodule.train_batch_sampler = None
        dataloader = cast(DataLoader, getattr(self.datamodule, f"{stage}_dataloader")(**dl_kwargs))
        dataset_encoder = DatasetEncoderRunner(model=self.encoder)
        self._encoder_runner.test(
            dataset_encoder,
            test_dataloaders=dataloader,
            verbose=False,
        )
        self.datamodule.train_batch_sampler = train_batch_sampler
        encodings, subgroup_inf, superclass_inf = dataset_encoder.encoded_dataset

        encodings = self.reducer.fit_transform(encodings)
        preds = self.clusterer(encodings)
        logging_dict = compute_metrics(
            preds=preds,
            subgroup_inf=subgroup_inf,
            superclass_inf=superclass_inf,
            prefix=stage,
            num_subgroups=self.datamodule.num_subgroups,
        )

        if isinstance(self.clusterer, Tomato) and self.clusterer.threshold == 1:
            pers_diagrams = {
                f"{stage}/pers_diagram_[thresh={self.clusterer.threshold}]": wandb.Image(
                    self.clusterer.plot()
                )
            }
            self.logger.experiment.log(pers_diagrams)
            plt.close("all")

        self.log_dict(logging_dict)

    def start(self, raw_config: dict[str, Any] | None = None):
        self.datamodule.setup()
        self.datamodule.prepare_data()
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        self.print(f"Current working directory: '{os.getcwd()}'")
        self.print(f"Artifacts directory created at: '{self.artifacts_dir.resolve()}'")

        logger = WandbLogger(
            entity="predictive-analytics-lab",
            project="topocluster",
            offline=self.log_offline,
            group=self.clusterer.__class__.__name__ if self.exp_group is None else self.exp_group,
        )
        hparams = {"artifacts_dir": self.artifacts_dir.resolve(), "cwd": os.getcwd()}
        if raw_config is not None:
            self.print("-----\n" + str(raw_config) + "\n-----")
            hparams.update(raw_config)
        logger.log_hyperparams(hparams)
        self.pretrainer.logger = logger
        self.trainer.logger = logger

        checkpointer_kwargs = dict(
            monitor="train/total_loss",
            dirpath=self.artifacts_dir,
            save_top_k=1,
            mode="max",
        )
        self.pretrainer.callbacks.append(
            ModelCheckpoint(**checkpointer_kwargs, filename="pretrain_best")
        )
        self.trainer.callbacks.append(ModelCheckpoint(**checkpointer_kwargs, filename="train_best"))

        # PRNG seeding
        pl.seed_everything(seed=self.seed)
        # Build the encoder
        self.encoder.build(self.datamodule)
        # Build the clusterer
        self.clusterer.build(encoder=self.encoder, datamodule=self.datamodule)
        # Pre-training phase
        self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        # Training phase
        if self.enc_freeze_depth:
            self.encoder.freeze(depth=self.enc_freeze_depth)
        # Build the sampler - the sampler is only used for joint training
        self.sampler.build(
            dataloader=self.datamodule.train_dataloader(shuffle=False), trainer=self.trainer
        )
        self.datamodule.train_batch_sampler = self.sampler
        self.trainer.fit(self, datamodule=self.datamodule)
        # Testing phase
        self.trainer.test(self, datamodule=self.datamodule)
        # Manually invoke exit for multirun compatibility
        logger.experiment.__exit__(None, 0, 0)


class DatasetEncoderRunner(pl.LightningModule):
    """Wrapper for extractor model."""

    encoded_dataset: Batch

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @implements(pl.LightningModule)
    def test_step(self, batch: Batch, batch_idx: int) -> Batch:
        return Batch(self(batch.x), *batch[1:])

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: list[Batch]) -> None:
        outputs_t = tuple(zip(*outputs))
        self.encoded_dataset = Batch(*(torch.cat(el, dim=0) for el in outputs_t))


