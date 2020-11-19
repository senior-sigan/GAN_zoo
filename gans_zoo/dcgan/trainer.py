from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from gans_zoo.dcgan.network import Discriminator, Generator, weights_init


class LitDCGAN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        beta1: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(nz=nz, ngf=ngf, nc=nc)
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(ndf=ndf, nc=nc)
        self.discriminator.apply(weights_init)

        self.real_label = 1.0
        self.fake_label = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input latent vector x
        :param x: latent vector of size (n_batches, z_dim, 1, 1)
        :return:
        """
        return self.generator.forward(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch[0]

        if optimizer_idx == 0:
            return self.generator_step(x)
        elif optimizer_idx == 1:
            return self.discriminator_step(x)

        msg = 'Expected optimizer_idx eq 0 or 1 but got {0}'
        raise AttributeError(msg.format(optimizer_idx))

    def generator_step(self, x: torch.Tensor):
        g_loss = self.generator_loss(x)
        self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x: torch.Tensor):
        d_loss = self.discriminator_loss(x)
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List]:
        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1

        opt_g = Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
        )
        opt_d = Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
        )

        return [opt_g, opt_d], []

    def generator_loss(self, x: torch.Tensor):
        batch_size = x.size(0)
        z = torch.randn(
            batch_size,
            *self.generator.input_shape,
            device=self.device,
        )
        y = torch.full((batch_size,), self.real_label, device=self.device)
        # For fake images discriminator should predict 0
        # But here we train discriminator to cheat generator
        # Thus we blame discriminator if it gets zeros from discriminator

        fakes = self.generator(z)

        D_output = self.discriminator(fakes).view(-1)
        g_loss = F.binary_cross_entropy(D_output, y)

        return g_loss

    def discriminator_loss(self, x_real: torch.Tensor):
        # train discriminator on real
        batch_size = x_real.size(0)
        y_real = torch.full((batch_size,), self.real_label, device=self.device)

        D_output = self.discriminator(x_real).view(-1)
        D_real_loss = F.binary_cross_entropy(D_output, y_real)

        # train discriminator on fake
        z = torch.randn(
            batch_size,
            *self.generator.input_shape,
            device=self.device,
        )
        x_fake = self(z)
        y_fake = torch.full((batch_size,), self.fake_label, device=self.device)

        D_output = self.discriminator(x_fake).view(-1)
        D_fake_loss = F.binary_cross_entropy(D_output, y_fake)

        D_loss = D_real_loss + D_fake_loss

        return D_loss
