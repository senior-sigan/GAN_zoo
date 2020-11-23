from gans_zoo.data.paired_data import PairedImagesFolderDataset
from gans_zoo.transforms.paired_transform import PairedTransform


def test_paired_dataset_find_all_images():
    dataset = PairedImagesFolderDataset(
        root='datasets/facades/train',
    )
    assert len(dataset) == 400
    dataset = PairedImagesFolderDataset(
        root='datasets/facades/test',
    )
    assert len(dataset) == 106
    dataset = PairedImagesFolderDataset(
        root='datasets/facades/val',
    )
    assert len(dataset) == 100


def test_paired_dataset_loads_pairs():
    transform = PairedTransform(
        crop_size=256,
        jitter=1.2,
    )
    dataset = PairedImagesFolderDataset(
        root='datasets/facades/train',
        transform=transform,
    )
    sample = dataset[0]
    input_img = sample['A']
    target_img = sample['B']
    assert input_img.size() == target_img.size()
