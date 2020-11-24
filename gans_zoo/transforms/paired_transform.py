from typing import TypeVar, List

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as funcs

TImage = TypeVar('TImage')


def apply_aug(sample, crop_params, flip_v, flip_h, resize_value, mean, std):
    sample = funcs.resize(
        img=sample,
        size=[resize_value, resize_value],
    )
    sample = funcs.crop(sample, *crop_params)
    if flip_h:
        sample = funcs.hflip(sample)
    if flip_v:
        sample = funcs.vflip(sample)

    sample = funcs.to_tensor(sample)
    sample = funcs.normalize(
        sample,
        mean=mean,
        std=std,
    )
    return sample


def apply_val_aug(sample, resize_value, mean, std):
    sample = funcs.resize(
        img=sample,
        size=[resize_value, resize_value],
    )

    sample = funcs.to_tensor(sample)
    sample = funcs.normalize(
        sample,
        mean=mean,
        std=std,
    )
    return sample


class PairedTransform(nn.Module):
    def __init__(self, crop_size: int, jitter: float):
        super().__init__()
        self.crop_size = crop_size
        self.resize_value = int(crop_size * jitter)
        self.flip_proba = 0.5
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        crop_params = transforms.RandomCrop.get_params(
            img=inputs[0],
            output_size=(self.crop_size, self.crop_size),
        )
        flip_v = torch.rand(1) < self.flip_proba
        flip_h = torch.rand(1) < self.flip_proba

        return [
            apply_aug(
                sample,
                crop_params,
                flip_v,
                flip_h,
                self.resize_value,
                self.mean,
                self.std,
            ) for sample in inputs
        ]


class PairedValTransform(nn.Module):
    def __init__(self, resize_value: int):
        super().__init__()
        self.resize_value = resize_value
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        return [
            apply_val_aug(sample, self.resize_value, self.mean, self.std)
            for sample in inputs
        ]
