"""Autoencoders"""
from __future__ import annotations
from abc import abstractmethod
from typing import cast

import pytorch_lightning as pl
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from topocluster.data.utils import Batch, ImageDims
from topocluster.layers.misc import View
from topocluster.utils.interface import implements


__all__ = ["AutoEncoder", "ConvAutoEncoder"]


class AutoEncoder(pl.LightningModule):
    """Base class for AutoEncoder models."""

    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim: int, lr: float = 1.0e-3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    @abstractmethod
    def _build(self, input_shape: int | ImageDims) -> tuple[nn.Module, nn.Module]:
        ...

    def build(self, input_shape: int | ImageDims) -> None:
        self.encoder, self.decoder = self._build(input_shape)

    @implements(nn.Module)
    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def get_loss(self, encoding: Tensor, x: Tensor, prefix: str = "") -> dict[str, Tensor]:
        if prefix:
            prefix += "/"
        return {f"{prefix}recon_loss": self.loss_fn(self.decoder(encoding), x)}

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.lr)

    @implements(pl.LightningModule)
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        x, _ = batch
        encoding = self.encoder(x)
        loss_dict = self.get_loss(encoding, x, prefix="train")
        total_loss = cast(Tensor, sum(loss_dict.values()))
        loss_dict["train/total_loss"] = total_loss
        self.logger.experiment.log(loss_dict)
        self.log_dict(loss_dict, prog_bar=True, logger=False)
        return total_loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, Tensor]:
        x, _ = batch
        encoding = self.encoder(x)
        loss_dict = self.get_loss(encoding, x, prefix="val")
        self.logger.experiment.log(loss_dict)
        self.log_dict(loss_dict, prog_bar=True, logger=False)
        return loss_dict


class ConvAutoEncoder(AutoEncoder):
    def __init__(
        self,
        init_hidden_dims: int,
        num_stages: int,
        latent_dim: int,
        lr: float = 1.0e-3,
    ):
        super().__init__(latent_dim=latent_dim, lr=lr)
        self.init_hidden_dims = init_hidden_dims
        self.num_stages = num_stages
        self.act = nn.GELU()

    def _down_conv(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            self.act,
        )

    def _up_conv(
        self,
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
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            self.act,
        )

    @implements(AutoEncoder)
    def _build(self, input_shape: ImageDims) -> tuple[nn.Sequential, nn.Sequential]:
        c_in, height, width = input_shape
        c_out = self.init_hidden_dims

        encoder_ls: list[nn.Module] = []
        decoder_ls: list[nn.Module] = []

        for level in range(self.num_stages):
            if level != 0:
                c_in = c_out
                c_out *= 2

            encoder_ls.append(
                nn.Sequential(
                    self._down_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                    self._down_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
                )
            )

            decoder_ls.append(
                nn.Sequential(
                    # inverted order
                    self._up_conv(
                        c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0
                    ),
                    self._down_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
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
