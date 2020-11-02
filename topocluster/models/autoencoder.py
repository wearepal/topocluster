"""Autoencoders"""
from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

from omegaconf.omegaconf import MISSING
import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

from topocluster.layers.misc import View


__all__ = ["AutoEncoder", "GatedConvAutoEncoder"]


INPUT_SHAPE = Tuple[int, int, int]


class AutoEncoder(pl.LightningModule):
    """Base class for AutoEncoder models."""

    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, lr: float = 1.0e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    def _build(self, input_shape: INPUT_SHAPE) -> Tuple[nn.Module, nn.Module]:
        ...

    def build(self, input_shape: INPUT_SHAPE) -> None:
        self.encoder, self.decoder = self._build(input_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> OrderedDict:
        x = batch[0] if isinstance(batch, Sequence) else batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        tqdm_dict = {"loss": loss}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output


class GatedConvAutoEncoder(AutoEncoder):
    def __init__(
        self,
        init_hidden_dims: int,
        levels: int,
        latent_dim: int,
        lr: float = 1.0e-3,
    ):
        super().__init__(lr=lr)
        self.init_hidden_dims = init_hidden_dims
        self.levels = levels
        self.latent_dim = latent_dim

    @staticmethod
    def _gated_down_conv(
        in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GLU(dim=1),
        )

    @staticmethod
    def _gated_up_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.GLU(dim=1),
        )

    def _build(self, input_shape: INPUT_SHAPE) -> Tuple[nn.Sequential, nn.Sequential]:
        encoder_ls: List[nn.Module] = []
        decoder_ls: List[nn.Module] = []
        c_in, height, width = input_shape
        c_out = self.init_hidden_dims

        for level in range(self.levels):
            if level != 0:
                c_in = c_out
                c_out *= 2

            encoder_ls.append(
                nn.Sequential(
                    self._gated_down_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                    self._gated_down_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
                )
            )

            decoder_ls.append(
                nn.Sequential(
                    # inverted order
                    self._gated_up_conv(
                        c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0
                    ),
                    self._gated_down_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
                )
            )

            height //= 2
            width //= 2

        flattened_size = c_out * height * width
        encoder_ls += [nn.Flatten()]
        encoder_ls += [nn.Linear(flattened_size, self.latent_dim)]

        decoder_ls += [View((c_out, height, width))]
        decoder_ls += [nn.Linear(self.latent_dim, flattened_size)]
        decoder_ls = decoder_ls[::-1]
        decoder_ls += [
            nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, stride=1, padding=0)
        ]

        encoder = nn.Sequential(*encoder_ls)
        decoder = nn.Sequential(*decoder_ls)

        return encoder, decoder
