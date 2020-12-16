import torch

from gans_zoo.pggan.loss import gan_loss


def test_gan_loss():
    logits = torch.randn(4, 1)
    loss = gan_loss(logits, True)
    print(loss.item())
