import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from gans_zoo.callbacks.cyclegan_tensorboard import TensorboardCycleGAN
from gans_zoo.cyclegan.trainer import LitCycleGAN
from gans_zoo.data.unpaired_data import UnpairedImagesFolderDataset
from gans_zoo.utils import norm_zero_one


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
    parser = LitCycleGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = LitCycleGAN.from_argparse_args(args)

    train_transform = transforms.Compose([
        transforms.Resize(int(model.input_size * args.jitter)),
        transforms.RandomCrop(model.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.CenterCrop(model.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = UnpairedImagesFolderDataset(
        root_a=os.path.join(args.train_dir, 'a'),
        root_b=os.path.join(args.train_dir, 'b'),
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_dataloaders = []
    if args.val_data_dir:
        val_ds = UnpairedImagesFolderDataset(
            root_a=os.path.join(args.val_data_dir, 'a'),
            root_b=os.path.join(args.val_data_dir, 'b'),
            transform=val_transform,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        val_dataloaders.append(val_loader)

    callbacks = [
        TensorboardCycleGAN(num_samples=3, normalize=norm_zero_one),
    ]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_dataloaders,
    )


if __name__ == '__main__':
    main()
