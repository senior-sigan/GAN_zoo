import torch


def norm_zero_one(tensor: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    min_ = float(tensor.min())
    max_ = float(tensor.max())
    tensor.clamp_(min=min_, max=max_)
    tensor.add_(-min_).div_(max_ - min_ + epsilon)
    return tensor
