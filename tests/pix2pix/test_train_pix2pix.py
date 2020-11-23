import pytorch_lightning as pl
from torch.utils.data import DataLoader

from gans_zoo.data.paired_data import FakePairedImagesDataset
from gans_zoo.pix2pix.trainer import LitPix2Pix


def test_fit():
    model = LitPix2Pix()
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)

    shape = (model.generator.out_channels, 256, 256)
    train_loader = DataLoader(FakePairedImagesDataset(shape), batch_size=1)
    val_loader = DataLoader(FakePairedImagesDataset(shape), batch_size=1)
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)
