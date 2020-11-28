import itertools
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import argparse_utils
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from gans_zoo.cyclegan.network import Discriminator, ResnetGenerator, \
    UNetGenerator, WeightsInit
from gans_zoo.cyclegan.scheduler import LinearLR


class LitCycleGAN(pl.LightningModule):
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
                            help='adam: optimizer beta1 parameter')
        parser.add_argument('--beta2', type=float, default=0.999,
                            help='adam: optimizer beta2 parameter')
        parser.add_argument('--lr', type=float, default=0.0002,
                            help='adam: learning rate')
        parser.add_argument('--lambda-cycle', type=float, default=10,
                            help='cycle loss weight')
        parser.add_argument('--lambda-identity', type=float, default=10,
                            help='identity loss weight')
        parser.add_argument('--decay-start-epoch', type=int, default=100,
                            help='epoch from which to start lr decay')
        parser.add_argument('--generator', type=str, default='resnet',
                            choices=['unet', 'resnet'], help='Generator type')
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser],
                           **kwargs) -> 'LitCycleGAN':
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    def __init__(
        self,
        input_size: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_identity: float = 10.0,
        lambda_cycle: float = 5.0,
        decay_start_epoch: int = 100,
        generator: str = 'resnet',
    ):
        super().__init__()
        self.save_hyperparameters()

        if generator == 'unet':
            self.generator_ab = UNetGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
            )
            self.generator_ba = UNetGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        elif generator == 'resnet':
            self.generator_ab = ResnetGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
            )
            self.generator_ba = ResnetGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            raise RuntimeError('Unknown generator {0}'.format(generator))

        self.discriminator_a = Discriminator(
            in_channels=in_channels,
        )
        self.discriminator_b = Discriminator(
            in_channels=in_channels,
        )
        self.generator_ab.apply(WeightsInit())
        self.generator_ba.apply(WeightsInit())
        self.discriminator_a.apply(WeightsInit())
        self.discriminator_b.apply(WeightsInit())

        self.real_label = 1.0
        self.fake_label = 0.0

        self.patch = Discriminator.patch_size(input_size, input_size)
        self.input_size = input_size
        self.img_dim = (in_channels, self.input_size, self.input_size)

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List]:
        lr = self.hparams.learning_rate
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2

        # Use SINGLE optimizer for two models!!
        # See original paper and implementation
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

        opt_g = Adam(
            itertools.chain(
                self.generator_ab.parameters(),
                self.generator_ba.parameters(),
            ),
            lr=lr,
            betas=(beta1, beta2),
        )
        opt_d_a = Adam(
            self.discriminator_a.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        opt_d_b = Adam(
            self.discriminator_b.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

        lr_scheduler_g = LinearLR(
            opt_g,
            self.trainer.max_epochs,
            self.hparams.decay_start_epoch,
            self.current_epoch,
        )
        lr_scheduler_d_a = LinearLR(
            opt_d_a,
            self.trainer.max_epochs,
            self.hparams.decay_start_epoch,
            self.current_epoch,
        )
        lr_scheduler_d_b = LinearLR(
            opt_d_b,
            self.trainer.max_epochs,
            self.hparams.decay_start_epoch,
            self.current_epoch,
        )

        optimizers = [opt_g, opt_d_a, opt_d_b]
        schedulers = [lr_scheduler_g, lr_scheduler_d_a, lr_scheduler_d_b]
        return optimizers, schedulers

    def forward(
        self,
        x: torch.Tensor,
        direction: str = 'ab',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates intermediate and reconstruction images from x.

        :param x: input image of shape (n_batches, n_channels, width, height)
        :param direction: direction in which execute cycleGAN: 'ab' or 'ba'
        :return:
        """
        if direction == 'ab':
            return self.forward_ab(x)
        elif direction == 'ba':
            return self.forward_ba(x)
        else:
            msg = 'Unknown direction {0}. Expected "ab" or "ba"'
            raise RuntimeError(msg.format(direction))

    def forward_ab(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fake_b = self.generator_ab(x)
        rec_a = self.generator_ba(fake_b)
        return fake_b, rec_a

    def forward_ba(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fake_a = self.generator_ba(x)
        rec_b = self.generator_ab(fake_a)
        return fake_a, rec_b

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        real_a, real_b = batch
        fake_a = self.generator_ba(real_b)
        fake_b = self.generator_ab(real_a)

        if optimizer_idx == 0:
            return self.generator_loss(real_a, real_b, fake_a, fake_b)
        elif optimizer_idx == 1:
            return self.discriminator_loss(real_a, fake_a,
                                           self.discriminator_a,
                                           'a')
        elif optimizer_idx == 2:
            return self.discriminator_loss(real_b, fake_b,
                                           self.discriminator_b,
                                           'b')

        msg = 'Expected optimizer_idx eq 0, 1, or 2 but got {0}'
        raise AttributeError(msg.format(optimizer_idx))

    def generator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ):
        y_real = torch.tensor(self.real_label, device=self.device)

        # Identity loss
        loss_identity_a = self.criterion_identity(
            self.generator_ba(real_a),
            real_a,
        )
        loss_identity_b = self.criterion_identity(
            self.generator_ab(real_b),
            real_b,
        )
        loss_identity = self.hparams.lambda_identity * (
            loss_identity_a + loss_identity_b) / 2

        # GAN loss
        d_fake_b_out = self.discriminator_b(fake_b)
        loss_gan_ab = self.criterion_GAN(
            d_fake_b_out,
            y_real.expand_as(d_fake_b_out),
        )
        d_fake_a_out = self.discriminator_a(fake_a)
        loss_gan_ba = self.criterion_GAN(
            d_fake_a_out,
            y_real.expand_as(d_fake_a_out),
        )
        loss_gan = (loss_gan_ab + loss_gan_ba) / 2

        # Cycle loss
        rec_a = self.generator_ba(fake_b)
        rec_b = self.generator_ab(fake_a)
        loss_cycle_a = self.criterion_cycle(real_a, rec_a)
        loss_cycle_b = self.criterion_cycle(real_b, rec_b)
        loss_cycle = self.hparams.lambda_cycle * (
            loss_cycle_a + loss_cycle_b) / 2

        loss = loss_gan + loss_cycle + loss_identity

        self.log_dict({
            'g_loss': loss,
            'gan_loss': loss_gan,
            'id_loss': loss_identity,
            'cycle_loss': loss_cycle,
        }, on_epoch=True, prog_bar=True)
        return loss

    def discriminator_loss(
        self,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
        discriminator: nn.Module,
        name: str,
    ):
        y_real = torch.tensor(self.real_label, device=self.device)
        y_fake = torch.tensor(self.fake_label, device=self.device)

        d_real_out = discriminator(real_x)
        loss_real = self.criterion_GAN(
            d_real_out,
            y_real.expand_as(d_real_out),
        )
        # TODO: get fake from a ImagePool like in official implementation
        d_fake_out = discriminator(fake_x.detach())
        loss_fake = self.criterion_GAN(
            d_fake_out,
            y_fake.expand_as(d_fake_out),
        )

        loss = (loss_real + loss_fake) / 2

        self.log_dict({
            f'd_{name}_loss': loss,
            f'real_{name}_loss': loss_real,
            f'fake_{name}_loss': loss_fake,
        }, on_epoch=True, prog_bar=True)
        return loss
