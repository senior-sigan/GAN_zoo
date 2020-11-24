from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from gans_zoo.callbacks.paired_image_sampler import \
    TensorboardPairedImageSampler
from gans_zoo.data.gan_data import ImagesFolder
from gans_zoo.pix2pix.trainer import LitPix2Pix
from gans_zoo.transforms.paired_transform import PairedTransform, \
    PairedValTransform


def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--train-data-dir', type=str, required=True)
    parser.add_argument('--val-data-dir', type=str)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of Data Loader workers')
    parser.add_argument('--jitter', type=float, default=1.2,
                        help='Jitter for random resize: input_size * jitter')
    return parser


def main():
    parser = ArgumentParser()
    parser = add_data_specific_args(parser)
    parser = LitPix2Pix.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = LitPix2Pix.from_argparse_args(args)

    transform_train = PairedTransform(
        crop_size=model.input_size,
        jitter=args.jitter,
    )
    transform_val = PairedValTransform(
        resize_value=model.input_size,
    )

    train_ds = ImagesFolder(
        root=args.train_data_dir,
        transform=transform_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_dataloaders = []
    if args.val_data_dir:
        val_ds = ImagesFolder(
            root=args.val_data_dir,
            transform=transform_val,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        val_dataloaders.append(val_loader)

    callbacks = [
        TensorboardPairedImageSampler(num_samples=3)
    ]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_dataloaders,
    )


if __name__ == '__main__':
    main()
