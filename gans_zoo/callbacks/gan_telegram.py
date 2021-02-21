import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.utilities import rank_zero_only

from gans_zoo.utils import tensor_to_file_like_object
from telegram_logger.logger import TelegramLogger


class TelegramLoggerCallback(pl.Callback):
    def __init__(
        self,
        tg_logger: TelegramLogger,
        num_samples: int = 64,
    ) -> None:
        super().__init__()
        self.tg_logger = tg_logger
        self.num_samples = num_samples
        self.nrows = int(num_samples ** 0.5)
        self.image_size = (
            256 * self.nrows,
            256 * self.num_samples // self.nrows,
        )

    @rank_zero_only
    def on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)
            pl_module.train()

        grid = torchvision.utils.make_grid(images, nrow=self.nrows)
        str_title = f'{pl_module.__class__.__name__}_images_{trainer.current_epoch}'
        image_file = tensor_to_file_like_object(grid, img_size=self.image_size)
        self.tg_logger.write_image(image_file, caption=str_title)
