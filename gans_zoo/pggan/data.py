from torchvision.transforms import transforms

from gans_zoo.data.cache_loader import SharedMemoryCacheLoader
from gans_zoo.data.gan_data import ImagesFolder


def build_dataset(root: str, image_size: int):
    load_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.3)),
    ])
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomCrop(image_size),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return ImagesFolder(
        root=root,
        loader=SharedMemoryCacheLoader(
            transform=load_transform,
        ),
        transform=transform,
    )
