"""Autoencoders"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from topocluster.layers.misc import View


__all__ = ["AutoEncoder", "build_conv_autoencoder", "build_fc_autoencoder"]


class AutoEncoder(pl.LightningModule):
    """Classical AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1.0e-3,
        recon_loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.recon_loss_fn = recon_loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch[0] if isinstance(batch, Sequence) else batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.recon_loss_fn(x_hat, x)
        self.log("train_loss", loss)
        tqdm_dict = {"loss": loss}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output


def gated_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GLU(dim=1),
    )


def gated_up_conv(in_channels, out_channels, kernel_size, stride, padding, output_padding):
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


def build_conv_autoencoder(
    input_shape: Sequence[int],
    init_hidden_dims: int,
    levels: int,
    latent_dim,
) -> Tuple[nn.Sequential, nn.Sequential]:
    encoder_ls: List[nn.Module] = []
    decoder_ls: List[nn.Module] = []
    c_in, height, width = input_shape
    c_out = init_hidden_dims

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        encoder_ls.append(
            nn.Sequential(
                gated_conv(c_in, c_out, kernel_size=3, stride=1, padding=1),
                gated_conv(c_out, c_out, kernel_size=4, stride=2, padding=1),
            )
        )

        decoder_ls.append(
            nn.Sequential(
                # inverted order
                gated_up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0),
                gated_conv(c_out, c_in, kernel_size=3, stride=1, padding=1),
            )
        )

        height //= 2
        width //= 2

    flattened_size = c_out * height * width
    encoder_ls += [nn.Flatten()]
    encoder_ls += [nn.Linear(flattened_size, latent_dim)]

    decoder_ls += [View((c_out, height, width))]
    decoder_ls += [nn.Linear(latent_dim, flattened_size)]
    decoder_ls = decoder_ls[::-1]
    decoder_ls += [nn.Conv2d(input_shape[0], input_shape[0], kernel_size=1, stride=1, padding=0)]

    encoder = nn.Sequential(*encoder_ls)
    decoder = nn.Sequential(*decoder_ls)

    return encoder, decoder


def _linear_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def build_fc_autoencoder(
    input_dims: int,
    init_hidden_dims: int,
    levels: int,
    latent_dim: int,
) -> Tuple[nn.Sequential, nn.Sequential]:
    encoder = []
    decoder = []

    c_in = input_dims
    c_out = init_hidden_dims

    for _ in range(levels):
        encoder += [_linear_block(c_in, c_out)]
        decoder += [_linear_block(c_out, c_in)]
        c_in = c_out

    encoder += [_linear_block(c_in, latent_dim)]
    decoder += [_linear_block(latent_dim, c_in)]
    decoder = decoder[::-1]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    return encoder, decoder
