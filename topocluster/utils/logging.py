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
            img = batch.x.to(pl_module.device)

            to_log = self._denormalize(img)

            str_title = f"{name}/{pl_module.__class__.__name__}"
            if isinstance(pl_module, AutoEncoder):
                with torch.no_grad():
                    recons = pl_module.reconstruct(img).cpu()
                to_log = torch.cat([to_log[None], recons[None]], dim=0).flatten(
                    start_dim=0, end_dim=1
                )
                str_title += "_images_&_recons"
            else:
                str_title += "_images"

            grid = torchvision.utils.make_grid(
                tensor=to_log,
                nrow=self.nrow,
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


# class ProgressBar(ProgressBarBase):
#     r"""
#     This is the default progress bar used by Lightning. It prints to `stdout` using the
#     :mod:`tqdm` package and shows up to four different bars:

#     - **sanity check progress:** the progress during the sanity check run
#     - **main progress:** shows training + validation progress combined. It also accounts for
#       multiple validation runs during training when
#       :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
#     - **validation progress:** only visible during validation;
#       shows total progress over all validation datasets.
#     - **test progress:** only active when testing; shows total progress over all test datasets.

#     For infinite datasets, the progress bar never ends.

#     If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
#     specific methods of the callback class and pass your custom implementation to the
#     :class:`~pytorch_lightning.trainer.trainer.Trainer`:

#     Example::

#         class LitProgressBar(ProgressBar):

#             def init_validation_tqdm(self):
#                 bar = super().init_validation_tqdm()
#                 bar.set_description('running validation ...')
#                 return bar

#         bar = LitProgressBar()
#         trainer = Trainer(callbacks=[bar])

#     Args:
#         refresh_rate:
#             Determines at which rate (in number of batches) the progress bars get updated.
#             Set it to ``0`` to disable the display. By default, the
#             :class:`~pytorch_lightning.trainer.trainer.Trainer` uses this implementation of the progress
#             bar and sets the refresh rate to the value provided to the
#             :paramref:`~pytorch_lightning.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
#             :class:`~pytorch_lightning.trainer.trainer.Trainer`.
#         process_position:
#             Set this to a value greater than ``0`` to offset the progress bars by this many lines.
#             This is useful when you have progress bars defined elsewhere and want to show all of them
#             together. This corresponds to
#             :paramref:`~pytorch_lightning.trainer.trainer.Trainer.process_position` in the
#             :class:`~pytorch_lightning.trainer.trainer.Trainer`.

#     """

#     def __init__(self, refresh_rate: int = 1, process_position: int = 0):
#         super().__init__()
#         self._refresh_rate = refresh_rate
#         self._process_position = process_position
#         self._enabled = True
#         self.main_progress_bar = None
#         self.val_progress_bar = None
#         self.test_progress_bar = None

#     def __getstate__(self):
#         # can't pickle the tqdm objects
#         state = self.__dict__.copy()
#         state["main_progress_bar"] = None
#         state["val_progress_bar"] = None
#         state["test_progress_bar"] = None
#         return state

#     @property
#     def refresh_rate(self) -> int:
#         return self._refresh_rate

#     @property
#     def process_position(self) -> int:
#         return self._process_position

#     @property
#     def is_enabled(self) -> bool:
#         return self._enabled and self.refresh_rate > 0

#     @property
#     def is_disabled(self) -> bool:
#         return not self.is_enabled

#     def disable(self) -> None:
#         self._enabled = False

#     def enable(self) -> None:
#         self._enabled = True

#     def init_sanity_tqdm(self) -> tqdm:
#         """ Override this to customize the tqdm bar for the validation sanity run. """
#         bar = tqdm(
#             desc="Validation sanity check",
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=False,
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )
#         return bar

#     def init_train_tqdm(self) -> tqdm:
#         """ Override this to customize the tqdm bar for training. """
#         bar = tqdm(
#             desc="Training",
#             initial=self.train_batch_idx,
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#             smoothing=0,
#         )
#         return bar

#     def init_predict_tqdm(self) -> tqdm:
#         """ Override this to customize the tqdm bar for predicting. """
#         bar = tqdm(
#             desc="Predicting",
#             initial=self.train_batch_idx,
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#             smoothing=0,
#         )
#         return bar

#     def init_validation_tqdm(self) -> tqdm:
#         """ Override this to customize the tqdm bar for validation. """
#         bar = tqdm(
#             desc="Validating",
#             position=(2 * self.process_position + 1),
#             disable=self.is_disabled,
#             leave=False,
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )
#         return bar

#     def init_test_tqdm(self) -> tqdm:
#         """ Override this to customize the tqdm bar for testing. """
#         bar = tqdm(
#             desc="Testing",
#             position=(2 * self.process_position),
#             disable=self.is_disabled,
#             leave=True,
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )
#         return bar

#     def on_sanity_check_start(self, trainer, pl_module):
#         super().on_sanity_check_start(trainer, pl_module)
#         self.val_progress_bar = self.init_sanity_tqdm()
#         reset(self.val_progress_bar, sum(trainer.num_sanity_val_batches))
#         self.main_progress_bar = tqdm(disable=True)  # dummy progress bar

#     def on_sanity_check_end(self, trainer, pl_module):
#         super().on_sanity_check_end(trainer, pl_module)
#         self.main_progress_bar.close()
#         self.val_progress_bar.close()

#     def on_train_start(self, trainer, pl_module):
#         super().on_train_start(trainer, pl_module)
#         self.main_progress_bar = self.init_train_tqdm()

#     def on_epoch_start(self, trainer, pl_module):
#         super().on_epoch_start(trainer, pl_module)
#         total_train_batches = self.total_train_batches
#         total_val_batches = self.total_val_batches
#         if total_train_batches != float("inf"):
#             # val can be checked multiple times per epoch
#             val_checks_per_epoch = total_train_batches // trainer.val_check_batch
#             total_val_batches = total_val_batches * val_checks_per_epoch
#         total_batches = total_train_batches + total_val_batches
#         reset(self.main_progress_bar, total_batches)
#         self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
#         if self._should_update(
#             self.train_batch_idx, self.total_train_batches + self.total_val_batches
#         ):
#             self._update_bar(self.main_progress_bar)
#             self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

#     def on_validation_start(self, trainer, pl_module):
#         super().on_validation_start(trainer, pl_module)
#         if not trainer.running_sanity_check:
#             self._update_bar(self.main_progress_bar)  # fill up remaining
#             self.val_progress_bar = self.init_validation_tqdm()
#             reset(self.val_progress_bar, self.total_val_batches)

#     def on_validation_batch_end(
#         self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#     ):
#         super().on_validation_batch_end(
#             trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#         )
#         if self._should_update(self.val_batch_idx, self.total_val_batches):
#             self._update_bar(self.val_progress_bar)
#             self._update_bar(self.main_progress_bar)

#     def on_validation_end(self, trainer, pl_module):
#         super().on_validation_end(trainer, pl_module)
#         self.main_progress_bar.set_postfix(trainer.progress_bar_dict)
#         self.val_progress_bar.close()

#     def on_train_end(self, trainer, pl_module):
#         super().on_train_end(trainer, pl_module)
#         self.main_progress_bar.close()

#     def on_test_start(self, trainer, pl_module):
#         super().on_test_start(trainer, pl_module)
#         self.test_progress_bar = self.init_test_tqdm()
#         self.test_progress_bar.total = convert_inf(self.total_test_batches)

#     def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
#         if self._should_update(self.test_batch_idx, self.total_test_batches):
#             self._update_bar(self.test_progress_bar)

#     def on_test_end(self, trainer, pl_module):
#         super().on_test_end(trainer, pl_module)
#         self.test_progress_bar.close()

#     def on_predict_start(self, trainer, pl_module):
#         super().on_predict_start(trainer, pl_module)
#         self.predict_progress_bar = self.init_predict_tqdm()
#         self.predict_progress_bar.total = convert_inf(self.total_predict_batches)

#     def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
#         if self._should_update(self.predict_batch_idx, self.total_predict_batches):
#             self._update_bar(self.predict_progress_bar)

#     def on_predict_end(self, trainer, pl_module):
#         self.predict_progress_bar.close()

#     def _should_update(self, current, total):
#         return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

#     def _update_bar(self, bar):
#         """ Updates the bar by the refresh rate without overshooting. """
#         if bar.total is not None:
#             delta = min(self.refresh_rate, bar.total - bar.n)
#         else:
#             # infinite / unknown size
#             delta = self.refresh_rate
#         if delta > 0:
#             bar.update(delta)


# def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
#     """ The tqdm doesn't support inf values. We have to convert it to None. """
#     if x == float("inf"):
#         return None
#     return x


# def reset(bar: tqdm, total: Optional[int] = None) -> None:
#     """ Resets the tqdm bar to 0 progress with a new total, unless it is disabled. """
#     if not bar.disable:
#         bar.reset(total=convert_inf(total))
