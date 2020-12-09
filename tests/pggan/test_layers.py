import pytest
import torch

from gans_zoo.pggan.layers import EqualizedConv2d, \
    get_layer_normalization_factor
from gans_zoo.pggan.network import Generator


def test_conv2d_equalized():
    layer = EqualizedConv2d(3, 8, kernel_size=3, padding=1)
    sample = torch.rand(1, 3, 16, 16)
    out = layer(sample)
    assert out is not None
    assert out.shape == (1, 8, 16, 16)


def test_get_norm_factor():
    layer = torch.nn.Conv2d(4, 8, kernel_size=3)
    factor = get_layer_normalization_factor(layer)
    assert factor == (2.0 / (4 * 3 * 3)) ** 0.5


def test_generator():
    gen = Generator(nz=32, depth_scale_0=8, nc=3)
    gen.alpha = 0

    z = torch.randn(2, 32)
    out = gen.forward(z)
    assert out.shape == (2, 3, 4, 4)


@pytest.mark.parametrize('alpha', [0, 0.9])
def test_generator_add_layers(alpha: float):
    gen = Generator(nz=32, depth_scale_0=8, nc=3)
    gen.alpha = 0

    z = torch.randn(2, 32)
    out = gen.forward(z)
    assert out.shape == (2, 3, 4, 4)
    assert out.shape == (2, 3, *gen.output_size)

    gen.alpha = alpha
    gen.add_layer(8)
    out = gen.forward(z)
    assert out.shape == (2, 3, 8, 8)
    assert out.shape == (2, 3, *gen.output_size)

    gen.add_layer(8)
    out = gen.forward(z)
    assert out.shape == (2, 3, 16, 16)
    assert out.shape == (2, 3, *gen.output_size)

    gen.add_layer(4)
    out = gen.forward(z)
    assert out.shape == (2, 3, 32, 32)
    assert out.shape == (2, 3, *gen.output_size)

    gen.add_layer(4)
    out = gen.forward(z)
    assert out.shape == (2, 3, 64, 64)
    assert out.shape == (2, 3, *gen.output_size)
