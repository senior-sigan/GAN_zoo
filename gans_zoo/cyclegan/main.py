import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from gans_zoo.callbacks.cyclegan_telegram import TelegramLoggerCallback
from gans_zoo.callbacks.cyclegan_tensorboard import TensorboardCycleGAN
from gans_zoo.callbacks.unpaired_sampler import UnpairedGridGenerator
from gans_zoo.cyclegan.trainer import LitCycleGAN
from gans_zoo.data.unpaired_data import UnpairedImagesFolderDataset
from gans_zoo.utils import norm_zero_one
from telegram_logger.logger import TelegramLogger


def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of Data Loader workers',
    )
    parser.add_argument(
        '--load_size', type=int, default=286,
        help='scale loaded images and then crop to a smaller size',
    )
    parser.add_argument(
        '--tg-token', type=str, required=False,
        help='Telegram bot token. Used to send epoch results to a chat',
    )
    parser.add_argument(
        '--tg-chat-id', type=int, required=False,
        help='Chat where to post epoch results',
    )
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
        transforms.Resize(args.load_size),
        transforms.RandomRotation(degrees=180),
        transforms.RandomCrop(model.input_size),
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
        root_a=os.path.join(args.data_dir, 'trainA'),
        root_b=os.path.join(args.data_dir, 'trainB'),
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_dataloaders = []
    test_a_path = os.path.join(args.data_dir, 'testA')
    test_b_path = os.path.join(args.data_dir, 'testB')
    if os.path.exists(test_a_path) and os.path.exists(test_b_path):
        val_ds = UnpairedImagesFolderDataset(
            root_a=test_a_path,
            root_b=test_b_path,
            transform=val_transform,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        val_dataloaders.append(val_loader)

    grid_generator = UnpairedGridGenerator(normalize=norm_zero_one)
    callbacks = [
        TensorboardCycleGAN(grid_generator),
    ]
    if args.tg_token is not None:
        tg_logger = TelegramLogger(
            token=args.tg_token,
            chat_id=args.tg_chat_id,
            module_name=__name__)
        callbacks += [TelegramLoggerCallback(grid_generator, tg_logger)]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_dataloaders,
    )


if __name__ == '__main__':
    main()
