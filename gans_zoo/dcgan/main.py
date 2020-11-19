from dataclasses import dataclass

import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, \
    TensorboardGenerativeModelImageSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from gans_zoo.data import ImagesFolder
from gans_zoo.dcgan.trainer import LitDCGAN


@dataclass
class Config:
    data_dir: str = "/Volumes/Media/datasets/crop-cosplayers"
    image_size: int = 64
    batch_size: int = 32
    workers: int = 1
    epochs: int = 100


def main():
    config = Config()
    pl.seed_everything(42)

    model = LitDCGAN()
    callbacks = [
        TensorboardGenerativeModelImageSampler(),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=callbacks,
        fast_dev_run=True,
    )

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImagesFolder(
        root=config.data_dir,
        transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.workers,
    )
    trainer.fit(model, train_dataloader=dataloader)


if __name__ == "__main__":
    main()
