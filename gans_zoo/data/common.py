import os
from glob import glob
from typing import List, Tuple, TypeVar, Union

TImage = TypeVar('TImage')
TTensor = TypeVar('TTensor')
TData = Union[TImage, TTensor]


def make_dataset(
    directory: str,
    extensions: Tuple[str, ...],
) -> List[str]:
    """
    Finds all images with extensions in the directory.

    All images are sorted in lexical order.
    :param directory:
    :param extensions:
    :return: list of found images
    """
    directory = os.path.expanduser(directory)
    instances = []
    for ext in extensions:
        mask = os.path.join(directory, '*{0}'.format(ext))
        for file_path in glob(mask):
            instances.append(file_path)
    return sorted(instances)
