import numpy as np
import torch
import groundingdino.datasets.transforms as T
from typing import Tuple
import PIL.Image


def load_image(image: PIL.Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize(
                [800], max_size=1333
            ),  # Short side will be 800, long side will be scaled with max 1333 pixels
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image, image_transformed
