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


def draw_samples(
    samples: List[List[torch.Tensor]],
    mode: str,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    normalize: Callable,
):
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

    str_title = f'{pl_module.__class__.__name__}_images_{mode}'
    trainer.logger.experiment.add_image(
        str_title,
        grid,
        global_step=trainer.global_step,
    )


class TensorboardCycleGAN(pl.Callback):
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

    def on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        datasets = {}

        if trainer.train_dataloader:
            datasets['train'] = trainer.train_dataloader.dataset

        if trainer.val_dataloaders:
            datasets['val'] = trainer.val_dataloaders[0].dataset

        assert len(datasets) > 0, \
            'Expected at least one dataset for samples generation'

        for mode, dataset in datasets.items():
            samples = random_sample(dataset, self.num_samples)
            draw_samples(samples, mode, trainer, pl_module, self.normalize)
