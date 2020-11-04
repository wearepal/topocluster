# """Main training file"""
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
from topocluster.data.data_modules import DataModule
from topocluster.models.autoencoder import AutoEncoder
from topocluster.optimisation.utils import count_occurances
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
        encoding = self.encoder(x)
        preds = self.clusterer(encoding)

        try:
            _counts = np.zeros((self.datamodule.num_classes,) * 2, dtype=np.int64)
            _counts, _ = count_occurances(_counts, preds.cpu().numpy(), s, y, s_count, args.cluster)
            _acc, _, _logging_dict_t = find_assignment(_counts, preds.size(0))
        except IndexError:
            _acc = float("nan")

        _ari = adjusted_rand_score(labels_true=ground_truth, labels_pred=_preds)
        _nmi = normalized_mutual_info_score(labels_true=ground_truth, labels_pred=_preds)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        labeled = y != -1

        encoding = self.encoder(x)
        recon_loss = self.encoder.get_loss(encoding)

        purity_loss = F.cross_entropy(self.clusterer.soft_labels[labeled], y[labeled])

        self.log("recon_loss", recon_loss, on_step=True, prog_bar=True, logger=True)
        self.log("purity_loss", purity_loss, on_step=True, prog_bar=True, logger=True)

        return recon_loss + purity_loss

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
