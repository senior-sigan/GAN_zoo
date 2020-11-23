import numpy as np

from gans_zoo.data.gan_data import ImagesFolder


def test_images_folder_finds_all_images():
    dataset = ImagesFolder(
        root='/Volumes/Media/datasets/anime',
    )
    assert len(dataset) == 24753


def test_images_folder_can_load_image():
    dataset = ImagesFolder(
        root='/Volumes/Media/datasets/anime',
    )
    img = dataset[0]
    assert img is not None
    img = np.asarray(img)
    print(img.ndim == 3)
