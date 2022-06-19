import os

import imageio.v2 as imageio
import numpy as np
from typing import List

BASE_PATH = "tests/resources/InputImages/"


def structure_output():
    pass


def load_images(image_names: List[str]):
    images = [
        imageio.imread(os.path.join(BASE_PATH, image_name))
        for image_name in image_names
    ]
    return images


def convert_gray_luminance(images: List[np.ndarray]):
    lgrays = list(
        map(
            lambda image: image.astype(np.uint8),
            map(
                lambda image: image[:, :, 0].astype(float) * 0.299
                + image[:, :, 1].astype(float) * 0.587
                + image[:, :, 2].astype(float) * 0.114,
                images,
            ),
        )
    )
    return lgrays


def binarize_images(images, T):
    for image in images:
        image[image < T] = 0
        image[image >= T] = 1
    return images

def erode_image(image):
    pass


def erode_images(images):
    eroded = [erode_image(img) for img in images]
    return eroded


def run():
    index = int(input().strip())
    Q = tuple(map(int, input().strip().split(" ")))
    F = int(input().strip())
    T = int(input().strip())
    B = int(input().strip())
    image_names = [str(input().strip()) for _ in range(B)]

    images = load_images(image_names)
    gray_images = convert_gray_luminance(images)
    bin_images = binarize_images(gray_images, T)


if __name__ == "__main__":
    run()
