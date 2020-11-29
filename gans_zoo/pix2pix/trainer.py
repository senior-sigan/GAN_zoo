from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import argparse_utils
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from gans_zoo.pix2pix.network import Discriminator, Generator, weights_init


class LitPix2Pix(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=3,
                            help='number of colors in the input image')
        parser.add_argument('--out_channels', type=int, default=3,
                            help='number of colors in the output image')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Adam\'s optimizer beta1 parameter')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='Adam\'s optimizer beta2 parameter')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='learning rate')
        parser.add_argument('--lambda-pixel', type=float, default=100,
                            help='from gan_loss + lambda * pixelwise_loss')
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser],
                           **kwargs) -> 'LitPix2Pix':
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    def __init__(
        self,
        input_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_pixel: float = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(
            in_channels=in_channels,
        )
        self.discriminator.apply(weights_init)

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        self.patch = Discriminator.patch_size(input_size, input_size)
        self.input_size = input_size
        self.img_dim = (in_channels, self.input_size, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input latent vector x
        :param x: latent vector of size (n_batches, z_dim, 1, 1)
        :return:
        """
        return self.generator(x)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        input_img, target_img = batch
        fake_img = self.generator(input_img)

        if optimizer_idx == 0:
            return self.generator_loss(input_img, target_img, fake_img)
        elif optimizer_idx == 1:
            return self.discriminator_loss(
                input_img,
                target_img,
                fake_img.detach(),
            )

        msg = 'Expected optimizer_idx eq 0 or 1 but got {0}'
        raise AttributeError(msg.format(optimizer_idx))

    def configure_optimizers(self) -> Tuple[List[Optimizer], List]:
        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2

        opt_g = Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        opt_d = Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

        return [opt_g, opt_d], []

    def generator_loss(
        self,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        fake_img: torch.Tensor,
    ):
        d_output = self.discriminator(input_img, fake_img)

        gan_loss = F.mse_loss(d_output, self.real_label.expand_as(d_output))
        pixelwise_loss = F.l1_loss(fake_img, target_img)

        g_loss = gan_loss + self.hparams.lambda_pixel * pixelwise_loss

        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_loss(
        self,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        fake_img: torch.Tensor,
    ):
        d_output = self.discriminator(input_img, target_img)
        d_real_loss = F.mse_loss(d_output, self.real_label.expand_as(d_output))

        d_output = self.discriminator(input_img, fake_img)
        d_fake_loss = F.mse_loss(d_output, self.fake_label.expand_as(d_output))

        d_loss = (d_real_loss + d_fake_loss) * 0.5

        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return d_loss
