import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from gans_zoo.data.cache_loader import SharedMemoryCacheLoader


def test_load_once():
    loads = 0

    def loader_mock(path: str) -> Image.Image:
        nonlocal loads
        loads += 1
        data = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        return Image.fromarray(data)

    loader = SharedMemoryCacheLoader(
        transform=transforms.Compose([
            transforms.Resize(32),
        ]),
        loader=loader_mock,
    )

    assert loads == 0
    loader("file.jpg")
    assert loads == 1
    loader("file.jpg")
    assert loads == 1
    loader("another.jpg")
    assert loads == 2
    loader("file.jpg")
    assert loads == 2
