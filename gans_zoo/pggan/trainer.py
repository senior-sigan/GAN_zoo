from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from gans_zoo.pggan.loss import gan_loss
from gans_zoo.pggan.network import Discriminator, Generator
from gans_zoo.utils import norm_zero_one


class LitPGGAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent-dim', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--nc', type=int, default=3,
                            help='number of colors in image')
        return parser

    def __init__(
        self,
        latent_dim: int = 512,
        depth_scale_0: int = 512,
        learning_rate: float = 0.001,
        beta1: float = 0,
        beta2: float = 0.99,
        nc: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            nz=latent_dim,
            depth_scale_0=depth_scale_0,
            nc=nc,
        )

        self.discriminator = Discriminator(
            depth_scale_0=depth_scale_0,
            nc=nc,
        )

        self.epoch_offset = 0
        self.n_batches = 0
        self.alphas = np.zeros(0)
        self.img_size = 4

        self.input_size = latent_dim

    @property
    def img_dim(self) -> Tuple[int, int, int]:
        return 3, self.img_size, self.img_size

    def grow(
        self, stage: str, scale: int, size: int, n_batches: int,
        n_epochs: int,
    ):
        self.img_size = size
        self.n_batches = n_batches
        self.epoch_offset += n_epochs

        if stage == 'stabilise':
            self.alphas = np.zeros(n_epochs * n_batches)
        elif stage == 'grow':
            self.alphas = np.linspace(1, 0, n_epochs * n_batches)
            self.generator.add_layer(scale)
            self.discriminator.add_layer(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input latent vector x
        :param x: latent vector of size (n_batches, z_dim, 1, 1)
        :return:
        """
        return norm_zero_one(self.generator.forward(x))

    def training_step(self, x_real, batch_idx, optimizer_idx):
        idx = batch_idx + (
            self.current_epoch - self.epoch_offset
        ) * self.n_batches
        alpha = self.alphas[idx]
        self.generator.alpha = alpha
        self.discriminator.alpha = alpha

        self.log('alpha', alpha, on_step=True, prog_bar=True)
        self.log('img_size', self.img_size, on_step=True, prog_bar=True)

        z = torch.randn(
            x_real.size(0),
            self.hparams.latent_dim,
            device=self.device,
        )
        x_fakes = self.generator(z)

        if optimizer_idx == 0:
            return self.generator_loss(x_fakes)
        elif optimizer_idx == 1:
            return self.discriminator_loss(x_real, x_fakes)

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

    def generator_loss(self, x_fake: torch.Tensor):
        D_output = self.discriminator(x_fake)
        g_loss = gan_loss(D_output, True)

        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor):
        D_output = self.discriminator(x_real)
        D_real_loss = gan_loss(D_output, True)

        D_output = self.discriminator(x_fake)
        D_fake_loss = gan_loss(D_output, False)

        D_loss = D_real_loss + D_fake_loss

        self.log('d_loss', D_loss, on_epoch=True, prog_bar=True)
        return D_loss
