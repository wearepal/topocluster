from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm
import wandb

from kit import implements
from topocluster.data.utils import Batch, NormalizationValues
from topocluster.models.autoencoder import AutoEncoder

__all__ = ["EncodingProgbar", "EmbeddingProgbar", "ImageLogger"]


class ImageLogger(pl.Callback):
    """Log Images."""

    def __init__(
        self,
        logging_freq: int = -1,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        scale_each: bool = False,
        pad_value: int = 1,
        norm_values: NormalizationValues | None = None,
    ) -> None:
        """Log images."""
        super().__init__()
        self.log_freq = logging_freq
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.norm_values = norm_values

    def _denormalize(self, img: Tensor) -> Tensor:
        if self.norm_values:
            img = img * torch.tensor(self.norm_values.std, device=img.device).view(
                1, -1, 1, 1
            ) + torch.tensor(self.norm_values.mean, device=img.device).view(1, -1, 1, 1)
        return img.clip(0, 1).cpu()

    def log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
        name: str,
    ) -> None:
        """Callback that logs images."""
        if trainer.logger is None:
            return
        if (self.log_freq == -1 and batch_idx == 1) or (
            self.log_freq > 0 and batch_idx % self.log_freq == 0
        ):
            imgs = batch.x.to(pl_module.device)[:self.nrow]

            to_log = self._denormalize(imgs)

            str_title = f"{name}/{pl_module.__class__.__name__}"
            if isinstance(pl_module, AutoEncoder):
                with torch.no_grad():
                    recons = self._denormalize(pl_module.reconstruct(imgs))
                to_log = torch.cat([to_log, recons], dim=0).flatten(
                    start_dim=0, end_dim=1
                )
                str_title += "_images_and_recons"
            else:
                str_title += "_images"

            grid = torchvision.utils.make_grid(
                tensor=to_log,
                nrow=imgs.size(0),
                padding=self.padding,
                normalize=self.normalize,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            str_title = f"{name}/{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.log(
                {str_title: wandb.Image(TF.to_pil_image(grid))},
                commit=False,
            )

    @implements(pl.Callback)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_images(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "train")

    @implements(pl.Callback)
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_images(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "val")


class EncodingProgbar(ProgressBar):
    """Custom Progress Bar for encoding."""

    def __init__(
        self, refresh_rate: int = 1, process_position: int = 0, trainer: pl.Trainer | None = None
    ):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self._trainer = trainer

    @implements(ProgressBar)
    def init_test_tqdm(self) -> tqdm:
        bar = super().init_test_tqdm()
        bar.set_description("Encoding dataset")
        return bar


class EmbeddingProgbar(ProgressBar):
    """Custom Progress Bar for embedding-generation."""

    def __init__(
        self, refresh_rate: int = 1, process_position: int = 0, trainer: pl.Trainer | None = None
    ):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self._trainer = trainer

    @implements(ProgressBar)
    def init_test_tqdm(self) -> tqdm:
        bar = super().init_test_tqdm()
        bar.set_description("Generating embeddings")
        return bar
