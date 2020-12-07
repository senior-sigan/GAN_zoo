import torch

from gans_zoo.pggan.layers import EqualizedConv2d, \
    get_layer_normalization_factor


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
