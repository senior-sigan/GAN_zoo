import torch
from torch import nn

"""
TODO: instead of deconvolution use conv+upscale.
See https://distill.pub/2016/deconv-checkerboard/
"""


def weights_init(m: nn.Module) -> None:
    """
    From the DCGAN paper, the authors specify that all model weights shall be
    randomly initialized from a Normal distribution with mean=0, stdev=0.02.

    Usage:
    > netG = Generator()
    > netG.apply(weights_init)

    :param m: module to apply custom weight initialisation
    :return:
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        """
        Neural network for generating images from a latent vector.
        Size of the generated image is 64x64.
        To make output bigger we need to add more upscale layers.

        :param nz: Size of z latent vector (i.e. size of generator input)
        :param ngf: Size of feature maps in generator
        :param nc: Number of channels in the training images. For color images this is 3
        """
        super().__init__()
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.input_shape = (nz, 1, 1)
        self.net = nn.Sequential(
            # input is Z of shape (nz, 1, 1), going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process tensor through the network.
        Generates image from the latent tensor.

        :param x: Latent vector of shape (n_batch, nz, 1, 1)
        :return: Tensor for generated images of shape (n_batch, 3, 64, 64)
        """
        return self.net.forward(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3) -> None:
        """
        Neural network to recognize fake or real images.
        Requires image of size 64x64 pixels.
        Return a probability of image being fake/real.

        :param ndf: Size of feature maps in discriminator
        :param nc: Number of channels in the training images. For color images this is 3
        """
        super().__init__()
        self.ndf = ndf
        self.nc = nc
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),  # TODO: why not softmax?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts whether image (x) is fake or real.
        x.size == (n_batches, C, H, W)

        :param x: batch of images
        :return: prediction (n_batches, 1, 1, 1)
        """
        return self.net.forward(x)
