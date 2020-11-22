import os
from glob import glob
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils import data
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


def make_dataset(
    directory: str,
    extensions: Tuple[str, ...],
) -> List[str]:
    directory = os.path.expanduser(directory)
    instances = []
    for ext in extensions:
        mask = os.path.join(directory, f"*{ext}")
        for file in glob(mask):
            instances.append(file)
    return list(sorted(instances))


class ImagesFolder(data.Dataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        if extensions is None:
            extensions = IMG_EXTENSIONS

        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            msg = "Found 0 files in folder of: {}\n".format(root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(
                    ",".join(extensions))
            raise RuntimeError(msg)

        self.root = root
        self.transform = transform
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __getitem__(self, index: int) -> Any:
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.samples)


class PairedImagesFolderDataset(data.Dataset):
    def __init__(
        self, root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if extensions is None:
            extensions = IMG_EXTENSIONS

        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            msg = "Found 0 files in folder of: {}\n".format(root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(
                    ",".join(extensions))
            raise RuntimeError(msg)

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.samples[index]
        sample = self.loader(path)
        w = sample.width
        h = sample.height
        target_image = sample.crop((0, 0, w // 2, h))
        input_image = sample.crop((w // 2, 0, w, h))

        if self.transform is not None:
            input_image = self.transform(input_image)
        if self.target_transform is not None:
            target_image = self.target_transform(target_image)
        return input_image, target_image

    def __len__(self) -> int:
        return len(self.samples)


class FakeImagesDataset(data.Dataset[torch.Tensor]):
    def __init__(self, shape: Tuple[int, ...], size: int = 16) -> None:
        self.shape = shape
        self.size = size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.randn(self.shape)

    def __len__(self) -> int:
        return self.size
