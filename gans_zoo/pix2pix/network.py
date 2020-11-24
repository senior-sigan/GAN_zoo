from typing import Optional, Tuple

import torch
from torch import nn


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DownScale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 normalize: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UpScale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        UNet-like model. Do pix2pix transformation.
        Input and output have same shape.
        Minimum input shape is 256x256.
        Model could be trained on large images without modifications
        because it's fully convolutional.

        :param in_channels: for color image 3
        :param out_channels: for color image 3
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_stack = [
            DownScale(in_channels, 64, normalize=False),
            DownScale(64, 128),
            DownScale(128, 256),
            DownScale(256, 512, dropout=0.5),
            DownScale(512, 512, dropout=0.5),
            DownScale(512, 512, dropout=0.5),
            DownScale(512, 512, dropout=0.5),
            DownScale(512, 512, normalize=False, dropout=0.5),
        ]

        self.up_stack = [
            UpScale(512, 512, dropout=0.5),
            UpScale(1024, 512, dropout=0.5),
            UpScale(1024, 512, dropout=0.5),
            UpScale(1024, 512, dropout=0.5),
            UpScale(1024, 256),
            UpScale(512, 128),
            UpScale(256, 64),
        ]

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        # reverse array order and skip last input,
        # because we don't need to concat same outputs
        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), 1)

        return self.final(x)


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Optional[str],
    ):
        super().__init__()

        # no need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_layer == 'instance_norm'

        layers = [nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
        )]

        if norm_layer == 'batch_norm':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_layer == 'instance_norm':
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        norm_layer: Optional[str] = 'instance_norm',
        ngf: int = 64
    ):
        """
        Patch-Discriminator.
        Returns BSx1x16x16 output (not BSx1 like usual Discriminator)

        :param in_channels: for color image 3
        :param norm_layer: None, instance_norm, batch_norm
        :param ngf: number of generator filters. By default is 64
        """
        super().__init__()
        self.model = nn.Sequential(
            DiscriminatorBlock(in_channels * 2, ngf, norm_layer=None),
            DiscriminatorBlock(ngf, ngf * 2, norm_layer=norm_layer),
            DiscriminatorBlock(ngf * 2, ngf * 4, norm_layer=norm_layer),
            DiscriminatorBlock(ngf * 4, ngf * 8, norm_layer=norm_layer),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf * 8, 1, kernel_size=4, padding=1, bias=False),
        )

    def forward(
        self,
        input_: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((input_, target), 1)
        return self.model(x)

    @staticmethod
    def patch_size(height: int, width: int) -> Tuple[int, int, int]:
        return 1, height // 2 ** 4, width // 2 ** 4
