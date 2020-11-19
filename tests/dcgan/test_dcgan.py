import torch

from gans_zoo.dcgan.network import Discriminator, Generator, weights_init


def test_generator_forward():
    gen = Generator()
    n_batches = 4
    batch = torch.rand(n_batches, gen.nz, 1, 1)
    out = gen.forward(batch)
    assert out.size() == (n_batches, 3, 64, 64)
    print(gen)


def test_discriminator_forward():
    dis = Discriminator()
    n_batches = 4
    batch = torch.rand(n_batches, dis.nc, 64, 64)
    out = dis.forward(batch)
    assert out.size() == (n_batches, 1, 1, 1)
    print(dis)


def test_generator_discriminator_flow():
    gen = Generator()
    dis = Discriminator()
    n_batches = 4
    batch = torch.rand(n_batches, gen.nz, 1, 1)

    fake = gen.forward(batch)
    label = dis.forward(fake)
    assert label.size() == (n_batches, 1, 1, 1)


def test_weight_init():
    gen = Generator()
    gen.apply(weights_init)

    dis = Discriminator()
    dis.apply(weights_init)
