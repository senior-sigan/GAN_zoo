from typing import Dict, TypeVar

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as funcs

TImage = TypeVar('TImage')


class PairedTransform(nn.Module):
    def __init__(self, crop_size: int, jitter: float):
        super().__init__()
        self.crop_size = crop_size
        self.resize_value = int(crop_size * jitter)
        self.flip_proba = 0.5
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def forward(self, inputs: Dict[str, TImage]) -> Dict[str, torch.Tensor]:
        sample = list(inputs.values())[0]

        crop_params = transforms.RandomCrop.get_params(
            img=sample,
            output_size=(self.crop_size, self.crop_size),
        )
        flip_v = torch.rand(1) < self.flip_proba
        flip_h = torch.rand(1) < self.flip_proba

        for k in inputs:
            inputs[k] = funcs.resize(
                img=inputs[k],
                size=[self.resize_value, self.resize_value],
            )
            inputs[k] = funcs.crop(inputs[k], *crop_params)
            if flip_h:
                inputs[k] = funcs.hflip(inputs[k])
            if flip_v:
                inputs[k] = funcs.vflip(inputs[k])

            inputs[k] = funcs.to_tensor(inputs[k])
            inputs[k] = funcs.normalize(
                inputs[k],
                mean=self.mean,
                std=self.std,
            )

        return inputs


class PairedValTransform(nn.Module):
    def __init__(self, resize_value: int):
        super().__init__()
        self.resize_value = resize_value
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def forward(self, inputs: Dict[str, TImage]) -> Dict[str, torch.Tensor]:
        for k in inputs:
            inputs[k] = funcs.resize(
                img=inputs[k],
                size=[self.resize_value, self.resize_value],
            )

            inputs[k] = funcs.to_tensor(inputs[k])
            inputs[k] = funcs.normalize(
                inputs[k],
                mean=self.mean,
                std=self.std,
            )

        return inputs
