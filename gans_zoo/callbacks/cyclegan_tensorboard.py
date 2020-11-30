import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from gans_zoo.callbacks.unpaired_sampler import UnpairedGridGenerator


class TensorboardCycleGAN(pl.Callback):
    def __init__(
        self,
        generator: UnpairedGridGenerator,
    ) -> None:
        super().__init__()
        self.generator = generator

    @rank_zero_only
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
            str_title = f'{pl_module.__class__.__name__}_images_{mode}'
            grid = self.generator.generate(pl_module, dataset)
            trainer.logger.experiment.add_image(
                str_title,
                grid,
                global_step=trainer.global_step,
            )
