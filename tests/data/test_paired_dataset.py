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
    dataset = PairedImagesFolderDataset(
        root="datasets/facades/train",
    )
    input_img, target_img = dataset[0]
    assert input_img.size == (256, 256)
    assert target_img.size == (256, 256)
    assert target_img != input_img
