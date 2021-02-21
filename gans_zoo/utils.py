import io
from typing import Optional, Tuple

import torch
from PIL.Image import NEAREST
from torchvision.transforms import functional as F


def norm_zero_one(tensor: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    min_ = float(tensor.min())
    max_ = float(tensor.max())
    tensor.clamp_(min=min_, max=max_)
    tensor.add_(-min_).div_(max_ - min_ + epsilon)
    return tensor


def tensor_to_file_like_object(
    tensor: torch.Tensor,
    img_size: Optional[Tuple[int, int]] = None,
):
    image = F.to_pil_image(tensor)
    if img_size is not None:
        image = image.resize(size=img_size, resample=NEAREST)
    file = io.BytesIO()
    image.save(file, "JPEG")
    file.seek(0)
    return file
