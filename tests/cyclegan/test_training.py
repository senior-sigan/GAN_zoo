import pytorch_lightning as pl
from torch.utils.data import DataLoader

from gans_zoo.cyclegan.trainer import LitCycleGAN
from gans_zoo.data.unpaired_data import FakeUnpairedImagesDataset


def test_fit():
    model = LitCycleGAN()
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)

    train_loader = DataLoader(FakeUnpairedImagesDataset(model.img_dim),
                              batch_size=1)
    val_loader = DataLoader(FakeUnpairedImagesDataset(model.img_dim),
                            batch_size=1)
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )
