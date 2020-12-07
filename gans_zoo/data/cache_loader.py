"""
Cache all the data during the first epoch to speed up training.
It saves in memory decoded images.
It reduces time to read image from the disk and decode jpg or png.
"""
import multiprocessing
from typing import Any, Callable, Optional

from PIL.Image import Image as PilImage
from torchvision.datasets.folder import default_loader


class SharedMemoryCacheLoader:
    """
    Load data using torchvision default_loader and cache data.

    This loader should be used with `torch.utils.data.DataLoader` if
    `num_workers` is greater than default `0`.
    """

    def __init__(
        self,
        transform: Optional[Callable[[PilImage], Any]] = None,
        loader: Callable[[str], PilImage] = default_loader,
    ):
        """
        Constructs a loader instance.

        :param transform: optional callable to transform image on first load.
            Could be used to resize image or convert to special color space.
        """
        self.loader = loader
        self.cache = multiprocessing.Manager().dict()
        self.transform = transform

    def __call__(self, path: str) -> Any:
        if path not in self.cache:
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            self.cache[path] = image
        return self.cache[path]
