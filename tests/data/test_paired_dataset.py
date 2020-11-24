from torch.utils.data.dataloader import DataLoader
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
    input_img, target_img = dataset[0]
    assert input_img.size() == target_img.size()


def test_paired_dataloader():
    transform = PairedTransform(
        crop_size=256,
        jitter=1.2,
    )
    dataset = PairedImagesFolderDataset(
        root='datasets/facades/train',
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
    )
    batch = next(iter(dataloader))
    img_a, img_b = batch
    assert img_a.shape == (2, 3, 256, 256)
    assert img_b.shape == (2, 3, 256, 256)
