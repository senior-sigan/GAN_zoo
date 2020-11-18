import torch

from gans_zoo.dcgan.network import Generator


def test_generator_forward():
    gen = Generator()
    n_batches = 4
    batch = torch.rand(n_batches, gen.z_dim, 1, 1)
    out = gen.forward(batch)
    assert out.size() == (n_batches, 3, 64, 64)
    print(gen)
