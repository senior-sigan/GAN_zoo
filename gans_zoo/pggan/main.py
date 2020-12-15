from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.callbacks import LatentDimInterpolator, \
    TensorboardGenerativeModelImageSampler

from gans_zoo.pggan.data import PGGANData
from gans_zoo.pggan.trainer import LitPGGAN

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


def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of Data Loader workers')
    return parser


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

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, )

    data = PGGANData(root=args.data_dir, batches=IMG_SIZE_TO_BATCH_SIZE)

    for size in model.sizes:
        print(f'Train model for size {size}')
        dataloader = data.next_loader()
        model.grow_gan(n_batches=len(dataloader))
        trainer.fit(model, train_dataloader=dataloader)


if __name__ == '__main__':
    main()
