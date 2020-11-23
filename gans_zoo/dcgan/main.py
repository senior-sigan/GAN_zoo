from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, \
    TensorboardGenerativeModelImageSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from gans_zoo.data.gan_data import ImagesFolder
from gans_zoo.dcgan.trainer import LitDCGAN


def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of Data Loader workers')
    return parser


def main():
    parser = ArgumentParser()
    parser = add_data_specific_args(parser)
    parser = LitDCGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = LitDCGAN()
    callbacks = [
        TensorboardGenerativeModelImageSampler(),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, )

    transform = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.CenterCrop(model.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImagesFolder(
        root=args.data_dir,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    trainer.fit(model, train_dataloader=dataloader)


if __name__ == '__main__':
    main()
