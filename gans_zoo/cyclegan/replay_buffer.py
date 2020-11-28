import random
from typing import List

import torch


class ReplayPool:
    def __init__(self, max_size: int = 50) -> None:
        assert max_size > 0, 'Replay pool must not be empty'
        self.max_size = max_size
        self.data: List[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        to_return = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if len(self.data) < self.max_size:
                self.data.append(image)
                to_return.append(image)
            else:
                # by 50% chance, the buffer will return a previously stored
                # image, and insert the current image into the buffer
                if random.uniform(0, 1) > 0.5:
                    # randint is inclusive
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = image
                else:
                    # by another 50% chance, the buffer will return
                    # the current image
                    to_return.append(image)
        # collect all the images and return
        return torch.cat(to_return, dim=0)
