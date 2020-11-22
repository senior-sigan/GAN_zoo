from torchvision import transforms
from torchvision.transforms import functional as funcs

from gans_zoo.aug_utils import build_trans
from gans_zoo.data import PairedImagesFolderDataset


def test_paired_dataset_find_all_images():
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/train",
    )
    assert len(dataset) == 400
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/test",
    )
    assert len(dataset) == 106
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/val",
    )
    assert len(dataset) == 100


def test_paired_dataset_loads_pairs():
    transform = transforms.Compose([
        build_trans(transforms.RandomRotation, funcs.rotate)(
            [-45, 45],
            expand=True
        ),
        build_trans(transforms.ToTensor, funcs.to_tensor)()
    ])
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/train",
        transform=transform,
    )
    sample = dataset[0]
    input_img = sample['A']
    target_img = sample['B']
    assert input_img.size() == target_img.size()


def test_paired_dataset_aug():
    transform = transforms.Compose([
        build_trans(transforms.Resize, funcs.resize)(size=300),
        build_trans(transforms.RandomCrop, funcs.crop, True)(256),
        build_trans(transforms.RandomHorizontalFlip, funcs.hflip)(),
        build_trans(transforms.RandomVerticalFlip, funcs.vflip)(),
        build_trans(transforms.ToTensor, funcs.to_tensor)(),
        build_trans(transforms.Normalize, funcs.normalize)((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5)),
    ])
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/train",
        transform=transform,
    )
    sample = dataset[0]
    input_img = sample['A']
    target_img = sample['B']
    assert input_img.size() == target_img.size()
