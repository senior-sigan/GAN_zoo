from typing import Callable, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import Dataset


def random_sample(
    dataset: Dataset,
    num_samples: int,
) -> List[List[torch.Tensor]]:
    idxs = torch.randperm(len(dataset), dtype=torch.int64)[:num_samples]
    return [dataset[idx] for idx in idxs]


def build_grid(samples: List[List[torch.Tensor]]) -> torch.Tensor:
    # grid visualisation, so torch.cat over 2nd dim
    # img_a1 - img_b1_fake - img_b1
    # img_a2 - img_b2_fake - img_b2
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
        img_a = img_a.to(device=pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            img_b_fakes = pl_module(img_a.unsqueeze(0))
            img_b_fake = img_b_fakes.squeeze(0)
            pl_module.train()
            imgs = [img_a, img_b_fake, img_b]
            imgs = [normalize(img.detach().cpu()) for img in imgs]
            images.append(imgs)

    grid = build_grid(images)

    str_title = f'{pl_module.__class__.__name__}_images_{mode}'
    trainer.logger.experiment.add_image(
        str_title,
        grid,
        global_step=trainer.global_step,
    )


def dummy_normalize(image):
    return image


class TensorboardPairedImageSampler(pl.Callback):
    def __init__(
        self,
        num_samples: int = 3,
        normalize: Optional[Callable] = None,
    ) -> None:
        """
        Generates images for paired dataset and logs to tensorboard.

        Your model must implement the forward function for generation.

        :param num_samples: number of images to generate
        """
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
