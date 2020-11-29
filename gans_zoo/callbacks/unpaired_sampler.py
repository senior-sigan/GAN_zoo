from typing import Callable, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


def random_sample(
    dataset: Dataset,
    num_samples: int,
) -> List[List[torch.Tensor]]:
    idxs = torch.randperm(len(dataset), dtype=torch.int64)[:num_samples]
    return [dataset[idx] for idx in idxs]


def dummy_normalize(image):
    return image


def build_grid(samples: List[List[torch.Tensor]]) -> torch.Tensor:
    # TODO: handle one-channel images
    line = []
    for sample in samples:
        img = torch.cat(sample, 2)
        line.append(img)
    return torch.cat(line, 1)


def generate_samples(
    samples: List[List[torch.Tensor]],
    pl_module: pl.LightningModule,
    normalize: Callable,
) -> torch.Tensor:
    images = []
    for sample in samples:
        img_a, img_b = sample
        with torch.no_grad():
            pl_module.eval()
            for direction, img in zip(['ab', 'ba'], [img_a, img_b]):
                fake, rec = pl_module(
                    img.unsqueeze(0).to(device=pl_module.device),
                    direction,
                )
                fake = fake.squeeze(0)
                rec = rec.squeeze(0)
                imgs = [img, fake, rec]
                imgs = [normalize(img.detach().cpu()) for img in imgs]
                images.append(imgs)
            pl_module.train()

    grid = build_grid(images)

    return grid


class UnpairedGridGenerator:
    def __init__(
        self,
        num_samples: int = 3,
        normalize: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.normalize = normalize
        if self.normalize is None:
            self.normalize = dummy_normalize

    def generate(self, pl_module: pl.LightningModule, dataset: Dataset):
        samples = random_sample(dataset, self.num_samples)
        generated = generate_samples(samples, pl_module, self.normalize)
        return generated
