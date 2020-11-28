import pytest
import torch
from torch import nn

from gans_zoo.cyclegan.network import ResnetBlock, ResnetGenerator


@pytest.mark.parametrize('padding_type', ['reflect', 'replicate', 'zeros'])
def test_resnet_block(padding_type):
    block = ResnetBlock(
        channels=4,
        padding_type=padding_type,
        norm_layer=nn.BatchNorm2d,
        use_dropout=True,
        use_bias=False,
    )
    shape = (2, 4, 16, 16)
    sample = torch.rand(size=shape)
    out = block(sample)
    assert out.shape == shape


@pytest.mark.parametrize('padding_type', ['reflect', 'replicate', 'zeros'])
def test_resnet_generator(padding_type):
    gen = ResnetGenerator(
        in_channels=3,
        out_channels=3,
        ngf=64,
        norm_layer_name='batch_norm',
        use_dropout=False,
        n_blocks=6,
        padding_type=padding_type,
    )
    print(gen)
    shape = (2, 3, 256, 256)
    sample = torch.rand(size=shape)
    out = gen(sample)
    assert out.shape == shape
