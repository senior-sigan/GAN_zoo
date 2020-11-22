import torch

from gans_zoo.pix2pix.network import DownScale, Generator, UpScale, weights_init


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
    inputs = torch.randn(size=(batch_size, 8, 16, 16))
    out_channels = 16
    layer = UpScale(
        in_channels=inputs.shape[1],
        out_channels=out_channels,
        dropout=0.5
    )

    output = layer.forward(inputs)
    assert output.size() == (
        inputs.shape[0],
        out_channels,
        inputs.shape[2] * 2,
        inputs.shape[3] * 2,
    )


def test_generator():
    batch_size = 2
    net = Generator(in_channels=1, out_channels=1)
    inputs = torch.randn(size=(batch_size, 1, 256, 256))
    output = net.forward(inputs)
    assert output.size() == inputs.size()
