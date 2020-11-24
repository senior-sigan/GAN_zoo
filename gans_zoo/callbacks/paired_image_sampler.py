from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import Dataset


def random_sample(dataset: Dataset, num_samples: int) -> List[
    Dict[str, torch.Tensor]
]:
    n = len(dataset)
    idxs = torch.randperm(n, dtype=torch.int64)[:num_samples]
    return [dataset[i] for i in idxs]


def build_grid(samples: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    # grid visualisation, so torch.cat over 2nd dim
    # img_a1 - img_b1_fake - img_b1
    # img_a2 - img_b2_fake - img_b2
    # TODO: handle one-channel images
    line = []
    for sample in samples:
        line.append(torch.cat((sample['A'], sample['B_fake'], sample['B']), 2))
    return torch.cat(line, 1)


def draw_samples(
    samples: List[Dict[str, torch.Tensor]],
    mode: str,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
):
    for sample in samples:
        img_a = sample['A'].to(device=pl_module.device)
        with torch.no_grad():
            pl_module.eval()
            img_b_fakes = pl_module(img_a.unsqueeze(0))
            sample['B_fake'] = img_b_fakes.squeeze(0)
            pl_module.train()

    grid = build_grid(samples)

    str_title = f'{pl_module.__class__.__name__}_images_{mode}'
    trainer.logger.experiment.add_image(
        str_title,
        grid,
        global_step=trainer.global_step,
    )


class TensorboardPairedImageSampler(pl.Callback):
    def __init__(self, num_samples: int = 3) -> None:
        """
        Generates images for paired dataset and logs to tensorboard.

        Your model must implement the forward function for generation.

        :param num_samples: number of images to generate
        """
        super().__init__()
        self.num_samples = num_samples

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
            draw_samples(samples, mode, trainer, pl_module)
