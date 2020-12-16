import torch

from gans_zoo.pggan.loss import gan_loss


def test_gan_loss_real():
    logits = torch.full(size=(4, 1), fill_value=100.0)
    loss = gan_loss(logits, True)
    assert loss.item() < 0.001


def test_gan_loss_fake():
    logits = torch.full(size=(4, 1), fill_value=-100.0)
    loss = gan_loss(logits, True)
    assert loss.item() > 4
