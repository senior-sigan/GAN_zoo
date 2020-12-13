import torch
from torch import nn


class MiniBatchStddev(nn.Module):
    def __init__(self, sub_group_size: int = 4):
        super().__init__()
        self.sub_group_size = sub_group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return minibatch_stddev_layer(x, self.sub_group_size)


def minibatch_stddev_layer(x, sub_group_size=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels
        3) expand the layer and concatenate it with the input
    Args:
        - x (tensor): previous layer
        - sub_group_size (int): size of the mini-batches on which the standard deviation
        should be computed
    """

    # TODO: refactor code more human understandable. Maybe named tensors?
    # https://github.com/tkarras/progressive_growing_of_gans/blob/35d6c23c578bdf2be185d026c6b3d366c1518120/networks.py#L127
    size = x.size()
    sub_group_size = min(size[0], sub_group_size)
    if size[0] % sub_group_size != 0:
        sub_group_size = size[0]
    G = int(size[0] / sub_group_size)
    if sub_group_size > 1:
        y = x.view(-1, sub_group_size, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, sub_group_size, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)
