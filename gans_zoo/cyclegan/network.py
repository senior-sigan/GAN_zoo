from functools import partial
from typing import Callable, Optional, Tuple, Type

import torch
from torch import nn


def get_norm_layer(name: str) -> Tuple[bool, Optional[Callable]]:
    if name == 'batch_norm':
        return False, nn.BatchNorm2d
    elif name == 'instance_norm':
        return True, nn.InstanceNorm2d
    else:
        return True, None


def get_activation(name: str) -> Type[Callable]:
    if name == 'leaky_relu':
        return partial(nn.LeakyReLU, negative_slope=0.2)
    elif name == 'relu':
        return nn.ReLU
    else:
        raise RuntimeError('Unknown activation {0}'.format(name))


class WeightsInit:
    def __init__(self, init_gain: float = 0.02) -> None:
        self.init_gain = init_gain

    def __call__(self, layer: nn.Module) -> None:
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            torch.nn.init.normal_(layer.weight.data, 0.0, self.init_gain)
            if hasattr(layer, 'bias') and layer.bias is not None:
                torch.nn.init.constant_(layer.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(layer.weight.data, 1.0, self.init_gain)
            torch.nn.init.constant_(layer.bias.data, 0.0)


class DownScale(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ) -> None:
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
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        skip_input: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        UNet-like Generator model.

        Do pix2pix transformation.
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

        self.down1 = DownScale(in_channels, 64, normalize=False)
        self.down2 = DownScale(64, 128)
        self.down3 = DownScale(128, 256)
        self.down4 = DownScale(256, 512, dropout=0.5)
        self.down5 = DownScale(512, 512, dropout=0.5)
        self.down6 = DownScale(512, 512, dropout=0.5)
        self.down7 = DownScale(512, 512, dropout=0.5)
        self.down8 = DownScale(512, 512, normalize=False, dropout=0.5)

        self.up1 = UpScale(512, 512, dropout=0.5)
        self.up2 = UpScale(1024, 512, dropout=0.5)
        self.up3 = UpScale(1024, 512, dropout=0.5)
        self.up4 = UpScale(1024, 512, dropout=0.5)
        self.up5 = UpScale(1024, 256)
        self.up6 = UpScale(512, 128)
        self.up7 = UpScale(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str,
        norm_layer: Optional[str],
        activation: Optional[str],
    ):
        super().__init__()

        use_bias, norm_layer_func = get_norm_layer(norm_layer)

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=use_bias,
            ),
        ]

        if norm_layer_func is not None:
            layers.append(norm_layer_func(out_channels))

        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str,
        norm_layer: Optional[str],
        activation: Optional[str],
    ):
        super().__init__()
        # Upsample instead of transposed conv
        # https://distill.pub/2016/deconv-checkerboard/

        use_bias, norm_layer_func = get_norm_layer(norm_layer)
        activation_func = get_activation(activation)

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=use_bias
            ),
            norm_layer_func(out_channels),
            activation_func(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        norm_layer: Optional[str] = 'batch_norm',
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
        conv_block = partial(
            ConvBlock,
            kernel_size=4,
            padding=1,
            padding_mode='zeros',
            activation='leaky_relu'
        )
        self.model = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=ngf, stride=2,
                       norm_layer=None),
            conv_block(in_channels=ngf, out_channels=ngf * 2, stride=2,
                       norm_layer=norm_layer),
            conv_block(in_channels=ngf * 2, out_channels=ngf * 4, stride=2,
                       norm_layer=norm_layer),
            conv_block(in_channels=ngf * 4, out_channels=ngf * 8, stride=2,
                       norm_layer=norm_layer),
            conv_block(in_channels=ngf * 8, out_channels=ngf * 8, stride=1,
                       norm_layer=norm_layer),
            nn.Conv2d(in_channels=ngf * 8, out_channels=1, kernel_size=4,
                      stride=1,
                      padding=1, bias=False),
        )

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_)

    @staticmethod
    def patch_size(height: int, width: int) -> Tuple[int, int, int]:
        return 1, height // 2 ** 4, width // 2 ** 4


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        channels: int,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
    ):
        """
        Construct the Resnet block.

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf


        :param channels: the number of channels in the conv layer.
        :param padding_type: the name of padding layer:
            reflect | replicate | zeros
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers.
        """
        super().__init__()

        use_bias, norm_layer_func = get_norm_layer(norm_layer)

        block = nn.Sequential()
        block.add_module('conv_1', nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            padding_mode=padding_type,
        ))
        block.add_module('norm_1', norm_layer_func(channels))
        block.add_module('activation', nn.ReLU(inplace=True))
        if use_dropout:
            block.add_module('dropout', nn.Dropout(0.5))
        block.add_module('conv_2', nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            padding_mode=padding_type,
        ))
        block.add_module('norm_2', norm_layer_func(channels))

        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks
    between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        norm_layer: str = 'batch_norm',
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = 'reflect',
    ) -> None:
        """
        Construct a Resnet-based generator.

        :param in_channels: the number of channels in input images
        :param out_channels: the number of channels in output images
        :param ngf: the number of filters in the last conv layer
        :param norm_layer: the name of normalization layer:
            batch_norm | instance_norm
        :param use_dropout: if use dropout layers
        :param n_blocks: the number of ResNet blocks
        :param padding_type: the name of padding layer in conv layers:
            reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        assert (in_channels > 0)
        assert (out_channels > 0)
        super().__init__()

        model = []
        model += [ConvBlock(
            in_channels=in_channels,
            out_channels=ngf,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode=padding_type,
            norm_layer=norm_layer,
            activation='relu',
        )]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ConvBlock(
                in_channels=ngf * mult,
                out_channels=ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='zeros',
                norm_layer=norm_layer,
                activation='relu',
            )]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(
                ngf * mult,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [UpConvBlock(
                in_channels=ngf * mult,
                out_channels=int(ngf * mult / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='zeros',
                norm_layer=norm_layer,
                activation='relu',
            )]
        model += [
            nn.Conv2d(
                in_channels=ngf,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                padding_mode=padding_type,
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward"""
        return self.model(x)
