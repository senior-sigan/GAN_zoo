import os
from typing import Callable, Optional, Tuple, TypeVar, Union

import torch
from torch.utils import data
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

from gans_zoo.data.common import make_dataset

TImage = TypeVar('TImage')
TTensor = TypeVar('TTensor')
TData = Union[TImage, TTensor]


class ImagesFolder(data.Dataset[TData]):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], TImage] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable[[TImage], TTensor]] = None,
    ) -> None:
        if extensions is None:
            extensions = IMG_EXTENSIONS

        root = os.path.abspath(root)
        samples = make_dataset(root, extensions)
        if not samples:
            msg = 'Found 0 files in folder of: {0}\n'.format(root)
            if extensions is not None:
                exts = ','.join(extensions)
                msg += 'Supported extensions are: {0}'.format(exts)
            raise RuntimeError(msg)

        print('Found {0} files in {1}'.format(len(samples), root))

        self.root = root
        self.transform = transform
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __getitem__(self, index: int) -> TData:
        """
        Get a sample from the dataset by index and apply transform if any.

        :param index:
        :return:
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        """
        Number of images in the dataset.

        :return:
        """
        return len(self.samples)


class FakeImagesDataset(data.Dataset[torch.Tensor]):
    def __init__(self, shape: Tuple[int, ...], size: int = 16) -> None:
        self.shape = shape
        self.size = size

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Generates a random sample of the shape.

        :param index:
        :return:
        """
        return torch.randn(self.shape)

    def __len__(self) -> int:
        """
        Number of images in the dataset.

        :return:
        """
        return self.size
