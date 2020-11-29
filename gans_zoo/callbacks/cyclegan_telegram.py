import pytorch_lightning as pl

from gans_zoo.callbacks.unpaired_sampler import UnpairedGridGenerator
from gans_zoo.utils import tensor_to_file_like_object
from telegram_logger.logger import TelegramLogger


class TelegramLoggerCallback(pl.Callback):
    def __init__(
        self,
        generator: UnpairedGridGenerator,
        tg_logger: TelegramLogger,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.tg_logger = tg_logger

    def on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        datasets = {}

        if trainer.train_dataloader:
            datasets['train'] = trainer.train_dataloader.dataset

        if trainer.val_dataloaders:
            datasets['val'] = trainer.val_dataloaders[0].dataset

        assert len(datasets) > 0, \
            'Expected at least one dataset for samples generation'

        for mode, dataset in datasets.items():
            name = pl_module.__class__.__name__
            step = trainer.global_step
            str_title = f'{name}_images_{mode} {step}'
            grid = self.generator.generate(pl_module, dataset)
            image_file = tensor_to_file_like_object(grid)
            self.tg_logger.write_image(image_file, caption=str_title)
