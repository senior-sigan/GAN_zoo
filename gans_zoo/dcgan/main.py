import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, \
    TensorboardGenerativeModelImageSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from gans_zoo.data import ImagesFolder
from gans_zoo.dcgan.trainer import LitDCGAN

data_dir = "/Volumes/Media/datasets/crop-cosplayers"
image_size = 64
batch_size = 32
workers = 1
epochs = 100


def main():
    pl.seed_everything(42)

    model = LitDCGAN()
    callbacks = [
        TensorboardGenerativeModelImageSampler(),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        fast_dev_run=True,
    )

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImagesFolder(
        root=data_dir,
        transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers,
    )
    trainer.fit(model, train_dataloader=dataloader)


if __name__ == "__main__":
    main()
