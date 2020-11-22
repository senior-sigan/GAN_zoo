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
        layers.append(nn.LeakyReLU(0.2))
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
        print(x.shape)
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
