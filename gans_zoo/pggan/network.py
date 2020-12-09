from typing import List, Tuple

import torch
from torch import nn

from gans_zoo.pggan.layers import EqualizedConv2d, EqualizedLinear, \
    PixelwiseNormalization


class Generator(nn.Module):
    def __init__(
        self,
        nz: int = 512,
        depth_scale_0: int = 512,
        nc: int = 3,
    ):
        super().__init__()
        self.nz = nz
        self.nc = nc
        self.scale_depths = [depth_scale_0]
        self.scale_layers: List[nn.Module] = [EqualizedConv2d(
            in_channels=self.scale_depths[0],
            out_channels=self.scale_depths[0],
            kernel_size=3,
            padding=1,
        )]

        # could be a stack with empty bottom and capacity = 2
        # because we always need only 2 or 1 to_rgb layers
        self.to_rgb = [self._to_rgb_block(self.scale_depths[0])]

        self.z_to_image = nn.Sequential(
            EqualizedLinear(
                in_features=nz,
                out_features=4 * 4 * self.scale_depths[0]
            ),
            nn.LeakyReLU(0.2),
        )

        self.pixelwise_norm = PixelwiseNormalization()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.lrely = nn.LeakyReLU(0.2)

        # Initialize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

    def add_layer(self, out_channels) -> None:
        """
        Adds next layer to the network with new out_channels

        Args:
            out_channels:
        """
        prev_out_channels = self.scale_depths[-1]
        self.scale_depths.append(out_channels)

        self.scale_layers.append(self._conv_block(
            in_channels=prev_out_channels,
            out_channels=out_channels,
        ))
        self.to_rgb.append(self._to_rgb_block(in_channels=out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input latent vector x through the network.

        Args:
            x: latent vector of size [batch_size, nz]

        Returns:
            Images tensor of size [batch_size, nc, width, height].
            Width and height you can get from .output_size property.
        """
        x = self.pixelwise_norm(x)
        x = self.z_to_image(x)
        x = x.view(-1, self.scale_depths[0], 4, 4)
        x = self.pixelwise_norm(x)
        for layer in self.scale_layers[:-1]:
            x = layer(x)

        x = self._skip_connect(x, self.scale_layers[-1])
        return x

    @property
    def output_size(self) -> Tuple[int, int]:
        side = int(4 * (2 ** (len(self.to_rgb) - 1)))
        return side, side

    def _to_rgb_block(self, in_channels: int) -> nn.Module:
        """
        Module to convert a tensor into RGB image.

        It's just a convolution layer with kernel=1 to set out_channels = nc.

        Args:
            in_channels: input image-tensor number of channels.
        """
        return EqualizedConv2d(
            in_channels=in_channels,
            out_channels=self.nc,
            kernel_size=1,
            padding=0,
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a block of convolution layers with activations and norms."""
        return nn.Sequential(
            self.upsample,
            EqualizedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            self.lrely,
            self.pixelwise_norm,
            EqualizedConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            self.lrely,
            self.pixelwise_norm,
        )

    def _skip_connect(
        self,
        x: torch.Tensor,
        block: nn.Module,
    ) -> torch.Tensor:
        """Apply a hack with smooth transition between new and old layers."""
        if self.alpha == 0:
            return self.to_rgb[-1](block(x))

        prev = self.upsample(self.to_rgb[-2](x))
        curr = self.to_rgb[-1](block(x))
        return (1 - self.alpha) * prev + self.alpha * curr


class Discriminator(nn.Module):
    pass
