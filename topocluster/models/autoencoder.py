"""Autoencoders"""
from __future__ import annotations
from abc import abstractmethod

from torch import Tensor
import torch.nn as nn

from kit import implements
from topocluster.data.datamodules import VisionDataModule
from topocluster.data.utils import Batch, ImageDims
from topocluster.layers.misc import View
from topocluster.models.base import Encoder


__all__ = ["AutoEncoder", "ConvAutoEncoder", "ConvAutoEncoderMNIST"]


class AutoEncoder(Encoder):
    """Base class for AutoEncoder models."""

    decoder: nn.Module

    def __init__(self, latent_dim: int, lr: float = 1.0e-3) -> None:
        super().__init__(lr=lr)
        self.latent_dim = latent_dim
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    @implements(Encoder)
    def _build(self, input_shape: int | ImageDims) -> tuple[nn.Module, nn.Module]:
        ...

    @implements(Encoder)
    def build(self, input_shape: int | ImageDims) -> None:
        self.encoder, self.decoder = self._build(input_shape)

    @implements(Encoder)
    def _get_loss(self, encoding: Tensor, batch: Batch) -> dict[str, Tensor]:
        recons = self.decoder(encoding)
        return {"recon_loss": self.loss_fn(recons, batch.x)}

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


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
    def _build(self, dm: VisionDataModule) -> tuple[nn.Sequential, nn.Sequential]:
        c_in, height, width = dm.dims
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
        decoder_ls += [nn.Conv2d(dm.dims.C, dm.dims.C, kernel_size=1, stride=1, padding=0)]

        encoder = nn.Sequential(*encoder_ls)
        decoder = nn.Sequential(*decoder_ls)

        return encoder, decoder


class ConvAutoEncoderMNIST(AutoEncoder):
    def __init__(
        self,
        latent_dim: int,
        lr: float = 1.0e-3,
    ):
        super().__init__(latent_dim=latent_dim, lr=lr)

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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

    def _build(self, dm: VisionDataModule) -> tuple[nn.Sequential, nn.Sequential]:
        encoder_ls: list[nn.Module] = [
            self._down_conv(
                in_channels=dm.dims.C, out_channels=32, kernel_size=5, stride=2, padding=2
            ),
            self._down_conv(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            self._down_conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
        ]
        encoder_ls.append(nn.Flatten())
        width = dm.dims.W // 2 ** 3
        height = dm.dims.H // 2 ** 3
        flattened_size = 128 * height * width
        encoder_ls.append(nn.Linear(flattened_size, self.latent_dim))

        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, flattened_size),
            View((128, height, width)),
            self._up_conv(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            self._up_conv(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            self._up_conv(
                in_channels=32,
                out_channels=dm.dims.C,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
        )
        return nn.Sequential(*encoder_ls), decoder
