import torch
from numpy import prod
from torch import nn


def get_layer_normalization_factor(layer: nn.Module) -> float:
    r"""
    Get He's constant for the given layer.

    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = layer.weight.size()
    fan_in = prod(size[1:])

    return (2.0 / fan_in) ** 0.5


class EqualizedLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        """A layer wrapper to set bias and weights.

        Set bias to zero.
        Set weights to be in the range (-1, 1).
        Apply He's normalization at runtime.

        Args:
            layer: a layer to wrap with initialization hacks
        """
        super().__init__()
        self.layer = layer
        self.layer.bias.data.fill_(0)
        self.layer.weight.data.normal_(0, 1)
        self.factor = get_layer_normalization_factor(self.layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input x through the layer and apply runtime normalization.

        Args:
            x: input tensor

        Returns:
            normalized tensor
        """
        return self.layer(x) * self.factor


class EqualizedConv2d(EqualizedLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        """Conv2d layer with equalized weights."""
        super().__init__(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        ))


class EqualizedLinear(EqualizedLayer):
    def __init__(self, in_features: int, out_features: int) -> None:
        """Linear layer with equalized weights."""
        super().__init__(nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        ))


class PixelwiseNormalization(nn.Module):
    """
    4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR
    A variant of “local response normalization” (Krizhevsky et al., 2012).
    With most datasets it does not change the results much, but it prevents
    the escalation of signal magnitudes very effectively when needed.
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: it should be / but here is *
        return x * (((x ** 2).mean(dim=1, keepdim=True) + self.epsilon).rsqrt())
