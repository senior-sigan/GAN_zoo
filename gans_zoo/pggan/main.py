from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, \
    TensorboardGenerativeModelImageSampler
from torch.utils.data import DataLoader

from gans_zoo.callbacks.gan_telegram import TelegramLoggerCallback
from gans_zoo.pggan.data import build_dataset
from gans_zoo.pggan.trainer import LitPGGAN
from telegram_logger.logger import TelegramLogger

IMG_SIZE_TO_BATCH_SIZE = {
    1024: 1,
    512: 2,
    256: 4,
    128: 8,
    64: 16,
    32: 32,
    16: 64,
    8: 128,
    4: 256,
}

SCALE_SIZES = [
    (512, 4),
    (512, 8),
    (512, 16),
    (512, 32),
    (256, 64),
    (128, 128),
    (64, 256),
    (32, 512),
    (16, 1024),
]

IMG_SIZE_TO_EPOCHS = {
    1024: 100,
    512: 100,
    256: 100,
    128: 250,
    64: 250,
    32: 500,
    16: 500,
    8: 250,
    4: 250,
}

STAGES = ['grow', 'stabilise']

STEPS = [
            (stage, scale, size)
            for scale, size in SCALE_SIZES
            for stage in STAGES
        ][1:]


def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of Data Loader workers')
    parser.add_argument(
        '--tg-token', type=str, required=False,
        help='Telegram bot token. Used to send epoch results to a chat',
    )
    parser.add_argument(
        '--tg-chat-id', type=int, required=False,
        help='Chat where to post epoch results',
    )
    return parser


def train(
    trainer: pl.Trainer,
    model: LitPGGAN,
    stage: str, scale: int, size: int, args,
):
    dataset = build_dataset(args.data_dir, size)
    dataloader = DataLoader(
        dataset,
        batch_size=IMG_SIZE_TO_BATCH_SIZE[size],
        shuffle=True,
        num_workers=args.workers,
    )

    print()
    print('-' * 80)
    print(
        f'Stage {stage}. Image size {size}. Scale {scale}. '
        f'Epochs {IMG_SIZE_TO_EPOCHS[size]}'
    )
    print('-' * 80)
    print()

    model.grow(
        stage, scale, size,
        n_batches=len(dataloader),
        n_epochs=IMG_SIZE_TO_EPOCHS[size],
    )

    trainer.max_epochs += IMG_SIZE_TO_EPOCHS[size]
    trainer.fit(model, train_dataloader=dataloader)
    trainer.current_epoch += 1  # in torch next fit starts on the previous epoch


def main():
    parser = ArgumentParser()
    parser = add_data_specific_args(parser)
    parser = LitPGGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = LitPGGAN()
    callbacks = [
        TensorboardGenerativeModelImageSampler(),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]
    if args.tg_token is not None:
        tg_logger = TelegramLogger(
            token=args.tg_token,
            chat_id=args.tg_chat_id,
            module_name=__name__)
        callbacks += [TelegramLoggerCallback(tg_logger)]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, )

    print(STEPS)
    trainer.max_epochs = 0
    for stage, scale, size in STEPS:
        train(trainer, model, stage, scale, size, args)


if __name__ == '__main__':
    main()
