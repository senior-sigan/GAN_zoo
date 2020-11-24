import torch


def norm_zero_one(t: torch.Tensor) -> torch.Tensor:
    min_ = float(t.min())
    max_ = float(t.max())
    t.clamp_(min=min_, max=max_)
    t.add_(-min_).div_(max_ - min_ + 1e-5)
    return t
