import torch

from gans_zoo.pix2pix.network import Discriminator, DownScale, Generator, \
    UpScale, weights_init


def test_apply_weights_init():
    net = Generator(in_channels=1, out_channels=1)
    net.apply(weights_init)


def test_down_scale():
    batch_size = 4
    inputs = torch.randn(size=(batch_size, 8, 16, 16))
    out_channels = 16
    layer = DownScale(
        in_channels=inputs.shape[1],
        out_channels=out_channels,
        normalize=True,
        dropout=0.5
    )

    output = layer.forward(inputs)
    assert output.size() == (
        inputs.shape[0],
        out_channels,
        inputs.shape[2] // 2,
        inputs.shape[3] // 2,
    )


def test_up_scale():
    batch_size = 4
    layer = UpScale(
        in_channels=4,
        out_channels=8,
        dropout=0.5,
    )
    inputs = torch.randn(size=(batch_size, 4, 16, 16))
    skips = torch.randn(size=(batch_size, 8, 32, 32))

    output = layer.forward(inputs, skips)
    assert output.size() == (
        batch_size,
        4 * 2 + 8,
        32,
        32,
    )


def test_generator():
    batch_size = 2
    net = Generator(in_channels=1, out_channels=1)
    inputs = torch.randn(size=(batch_size, 1, 256, 256))
    output = net.forward(inputs)
    assert output.size() == inputs.size()


def test_discriminator():
    batch_size = 2
    net = Discriminator(in_channels=3, norm_layer='instance_norm')
    input_ = torch.randn(size=(batch_size, 3, 256, 256))
    target = torch.randn(size=(batch_size, 3, 256, 256))
    out = net.forward(input_, target)

    patch = (batch_size, *Discriminator.patch_size(256, 256))
    assert out.size() == patch
    assert out.size() == (batch_size, 1, 16, 16)
