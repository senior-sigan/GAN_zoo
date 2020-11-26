from typing import Callable, List, Optional, Tuple, TypeVar, Union

import torch
from torch.utils import data
from torchvision.datasets.folder import default_loader

from gans_zoo.data.gan_data import ImagesFolder

TImage = TypeVar('TImage')
TTensor = TypeVar('TTensor')
TData = Union[TImage, TTensor]


class UnpairedImagesFolderDataset(data.Dataset[TData]):
    def __init__(
        self,
        root_a: str,
        root_b: str,
        loader: Callable[[str], TImage] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable[[TImage], TTensor]] = None,
    ) -> None:
        """
        Unpaired dataset from two folders.

        Warning! The dataset size is equal to the smallest dataset's size.
        Use IterDataset instead if you have different sized datasets.

        :param root_a:
        :param root_b:
        :param loader:
        :param extensions:
        :param transform:
        """
        self.dataset_a = ImagesFolder(root_a, loader, extensions, transform)
        self.dataset_b = ImagesFolder(root_b, loader, extensions, transform)

    def __getitem__(self, index: int) -> TData:
        """
        Get a sample from the dataset by index and apply transform if any.

        :param index:
        :return:
        """
        return self.dataset_a[index], self.dataset_b[index]

    def __len__(self) -> int:
        """
        Number of images in the dataset.

        :return:
        """
        return min(len(self.dataset_a), len(self.dataset_b))


class FakeUnpairedImagesDataset(data.Dataset[torch.Tensor]):
    def __init__(self, shape: Tuple[int, ...], size: int = 16) -> None:
        self.shape = shape
        self.size = size

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        """
        Generates a pair of random samples of the shape.

        :param index:
        :return: tuple of A and B tensor
        """
        return [torch.randn(self.shape), torch.randn(self.shape)]

    def __len__(self) -> int:
        """
        Number of images in the dataset.

        :return:
        """
        return self.size
