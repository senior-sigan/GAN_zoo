import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from gans_zoo.dcgan.trainer import LitDCGAN


def test_gen_loss():
    model = LitDCGAN()
    batch = torch.randn(4, model.generator.nc, 64, 64)
    loss = model.generator_loss(batch)
    print(loss)


def test_disc_loss():
    model = LitDCGAN()
    batch = torch.randn(4, model.discriminator.nc, 64, 64)
    loss = model.discriminator_loss(batch)
    print(loss)


def test_fit():
    model = LitDCGAN()
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)

    inputs = torch.randn(8, model.generator.nc, 64, 64)
    train_loader = DataLoader(TensorDataset(inputs), batch_size=1)
    val_loader = DataLoader(TensorDataset(inputs), batch_size=1)
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)
