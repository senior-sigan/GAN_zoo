import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from gans_zoo.callbacks.paired_image_sampler import \
    TensorboardPairedImageSampler, build_grid
from gans_zoo.data.paired_data import FakePairedImagesDataset
from gans_zoo.pix2pix.trainer import LitPix2Pix


def test_on_epoch_end():
    dataset = FakePairedImagesDataset(shape=(3, 256, 256))
    data_loader = DataLoader(dataset, batch_size=2)
    trainer = pl.Trainer(fast_dev_run=True)
    pl_module = LitPix2Pix()
    trainer.fit(
        pl_module,
        train_dataloader=data_loader,
        val_dataloaders=data_loader,
    )

    sampler = TensorboardPairedImageSampler()
    sampler.on_epoch_end(trainer, pl_module)


def test_build_grid():
    samples = [
        {
            'A': torch.full(size=(3, 4, 4), fill_value=1),
            'B_fake': torch.full(size=(3, 4, 4), fill_value=2),
            'B': torch.full(size=(3, 4, 4), fill_value=3),
        },
        {
            'A': torch.full(size=(3, 4, 4), fill_value=4),
            'B_fake': torch.full(size=(3, 4, 4), fill_value=5),
            'B': torch.full(size=(3, 4, 4), fill_value=6),
        }
    ]
    grid = build_grid(samples)
    assert grid.shape == (3, 4 * 2, 4 * 3)
    grid = torch.sum(grid, dim=0)
    assert grid[0][0] == 3
    assert grid[0][4] == 6
    assert grid[0][8] == 9
    assert grid[4][0] == 12
    assert grid[4][4] == 15
    assert grid[4][8] == 18
