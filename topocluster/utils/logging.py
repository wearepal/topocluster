from __future__ import annotations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

__all__ = ["EncodingProgbar", "EmbeddingProgbar", "ImageLogger", "visualize_clusters"]


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
        prefix: str,
    ) -> None:
        """Callback that logs images."""
        if trainer.logger is None:
            return
        if (self.log_freq == -1 and batch_idx == 1) or (
            self.log_freq > 0 and batch_idx % self.log_freq == 0
        ):
            imgs = batch.x.to(pl_module.device)[: self.nrow]

            to_log = self._denormalize(imgs)

            str_title = f"{prefix}/{pl_module.__class__.__name__}"
            if isinstance(pl_module, AutoEncoder):
                with torch.no_grad():
                    recons = self._denormalize(pl_module.reconstruct(imgs))
                to_log = torch.cat([to_log[None], recons[None]], dim=0).flatten(
                    start_dim=0, end_dim=1
                )
                str_title += "_images_&_recons"
            else:
                str_title += "_images"
            breakpoint()

            grid = torchvision.utils.make_grid(
                tensor=to_log,
                nrow=imgs.size(0),
                padding=self.padding,
                normalize=self.normalize,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            trainer.logger.experiment.log(
                {str_title: wandb.Image(TF.to_pil_image(grid))},
                commit=False,
            )

    @implements(pl.Callback)
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[Any],
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_images(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            prefix="train",
        )

    @implements(pl.Callback)
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[Any],
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.log_images(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            prefix="val",
        )


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


def visualize_clusters(encodings: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> plt.Figure:
    if isinstance(encodings, Tensor):
        encodings = encodings.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()
    if not encodings.ndim == 2 and encodings.shape[1] == 2:
        raise ValueError("Encodings must be 2-dimensional vectors.")
    cluster_viz, ax = plt.subplots(dpi=100)
    ax.scatter(encodings[:, 0], encodings[:, 1], c=labels, cmap="tab10")  # type: ignore[arg-type]
    ax.set_title("Cluster Visualization")
    return cluster_viz
