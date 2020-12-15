from typing import Dict

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from gans_zoo.data.cache_loader import SharedMemoryCacheLoader
from gans_zoo.data.gan_data import ImagesFolder


def build_dataset(root: str, image_size: int):
    load_transform = transforms.Compose([
        transforms.Resize(image_size * 1.3),
    ])
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomCrop(image_size),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = ImagesFolder(
        root=root,
        loader=SharedMemoryCacheLoader(
            transform=load_transform,
        ),
        transform=transform,
    )
    return ds


class PGGANData:
    def __init__(
        self,
        root: str,
        batches: Dict[int, int],
        num_workers: int = 8,
    ):
        self.root = root
        self.num_workers = num_workers
        self.sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.batches = batches
        self.iteration = 0

    def next_loader(self):
        size = self.sizes[self.iteration]
        dataset = build_dataset(
            self.root,
            size,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batches[size],
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.iteration += 1
        return loader
