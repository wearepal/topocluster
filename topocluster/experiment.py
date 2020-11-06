# """Main training file"""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.optim import Adam, Optimizer
from torch.tensor import Tensor

from topocluster.clustering.common import Clusterer
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

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        x, y = batch
        y_np = y.cpu().numpy()

        encoding = self.encoder(x)
        preds = self.clusterer(encoding).cpu().detach().numpy()

        metrics = {
            "val/ARI": adjusted_rand_score(labels_true=y_np, labels_pred=preds),
            "val/NMI": normalized_mutual_info_score(labels_true=y_np, labels_pred=preds),
        }

        self.logger.experiment.log(metrics)

        return metrics

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch

        encoding = self.encoder(x)
        self.clusterer.fit(encoding)

        loss_dict = self.encoder.get_loss(encoding, x, prefix="train")
        loss_dict.update(self.clusterer.get_loss(encoding, y, prefix="train"))

        self.logger.experiment.log(loss_dict)

        return sum(loss_dict.values())

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, float]:
        val_metrics = self.validation_step(batch, batch_idx)
        test_metrics = {"test/ARI": val_metrics["val/ARI"], "test/NMI": val_metrics["val/NMI"]}
        self.logger.experiment.log(test_metrics)

        return test_metrics

    def start(self, raw_config: Optional[Dict[str, Any]] = None):
        self.datamodule.setup()
        if self.hparams.use_wandb:
            logger = WandbLogger(entity="predictive-analytics-lab", project="topocluster")
            if raw_config is not None:
                logger.log_hyperparams(raw_config)
            self.pretrainer.logger = logger
            self.trainer.logger = logger

        pl.seed_everything(seed=self.hparams.seed)
        self.encoder.build(self.datamodule.dims)
        self.clusterer.build(self.encoder.latent_dim, self.datamodule.num_classes)

        self.pretrainer.fit(self.encoder, datamodule=self.datamodule)
        self.trainer.fit(self, datamodule=self.datamodule)
        self.trainer.test(self, datamodule=self.datamodule)
