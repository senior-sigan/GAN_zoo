import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        """
        Neural network fo generating images from a latent vector.
        Size of the generated image is 64x64.

        :param nz: Size of z latent vector (i.e. size of generator input)
        :param ngf: Size of feature maps in generator
        :param nc: Number of channels in the training images. For color images this is 3
        """
        super().__init__()
        self.z_dim = nz
        self.nc = nc
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process tensor through the network.
        :param x: Latent vector of shape (n_batch, nz, 1, 1)
        :return: Tensor for generated images of shape (n_batch, 3, 64, 64)
        """
        return self.net.forward(x)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass
