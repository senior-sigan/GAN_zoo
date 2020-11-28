import pytest
import torch

from gans_zoo.cyclegan.network import Discriminator, ResnetBlock, \
    ResnetGenerator


@pytest.mark.parametrize('padding_type', ['reflect', 'replicate', 'zeros'])
def test_resnet_block_paddings(padding_type):
    block = ResnetBlock(
        channels=4,
        padding_type=padding_type,
        norm_layer='batch_norm',
        use_dropout=True
    )
    shape = (2, 4, 16, 16)
    sample = torch.rand(size=shape)
    out = block(sample)
    assert out.shape == shape


@pytest.mark.parametrize('norm_layer', ['batch_norm', 'instance_norm'])
def test_resnet_block_norms(norm_layer):
    block = ResnetBlock(
        channels=4,
        padding_type='zeros',
        norm_layer=norm_layer,
        use_dropout=True
    )
    shape = (2, 4, 16, 16)
    sample = torch.rand(size=shape)
    out = block(sample)
    assert out.shape == shape


def test_resnet_generator():
    gen = ResnetGenerator(
        in_channels=3,
        out_channels=3,
        ngf=64,
        norm_layer='batch_norm',
        use_dropout=False,
        n_blocks=6,
        padding_type='reflect',
    )
    print(gen)
    shape = (2, 3, 256, 256)
    sample = torch.rand(size=shape)
    out = gen(sample)
    assert out.shape == shape


def test_path_gan_discriminator():
    net = Discriminator(
        in_channels=3,
        norm_layer='batch_norm',
        ngf=64,
    )
    print(net)
    sample = torch.rand(size=(2, 3, 256, 256))
    out = net(sample)
    assert out.shape == (2, 1, 14, 14)
