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
    eroded = np.empty_like(image)
    rows, cols = image.shape
    for i in range(1, rows):
        for j in range(1, cols):
            min_val = np.min(image[i - 1 : i + 1, j - 1 : j + 1])
            eroded[i, j] = min_val
    return eroded


def dilate_image(image):
    dilated = np.empty_like(image)
    rows, cols = image.shape
    for i in range(1, rows):
        for j in range(1, cols):
            max_val = np.max(image[i - 1 : i + 1, j - 1 : j + 1])
            dilated[i, j] = max_val
    return dilated


def erode_images(images):
    eroded = list(map(erode_image, images))
    return eroded


def dilate_images(images):
    dilated = list(map(dilate_image, images))
    return dilated


def closing(images):
    dilated = dilate_images(images)
    closed = erode_images(dilated)
    return closed


def opening(images):
    eroded = erode_images(images)
    opened = dilate_images(eroded)
    return opened


def masks_from(gray_images, morphed_images):
    masks = []
    for gray, morphed in zip(gray_images, morphed_images):
        mask1 = np.empty_like(gray)
        mask2 = np.empty_like(gray)
        mask1[morphed == 0] = gray[morphed == 0]
        mask2[morphed == 1] = gray[morphed == 1]
        masks.append((mask1, mask2))
    return masks


def cooccurrence_prob_matrix(mask, Q):
    row, col = mask.shape
    cooc_side = np.max(mask) + 1
    cooc = np.zeros((cooc_side, cooc_side))
    for i in range(1, row):
        for j in range(1, col):
            cooc[i, j][i + Q[0], j + Q[1]] += 1
    cooc /= np.sum(cooc)
    return cooc


def cooccurrence_prob_matrixes(masks, Q):
    matrixes = []
    for mm in masks:
        coocs = [cooccurrence_prob_matrix(m, Q) for m in mm]
        matrixes.append(tuple(coocs))
    return matrixes


def compute_haralick_descriptors(cooccurrences):
    descriptors = []
    for coocs in cooccurrences:
        auto_corr = [auto_correlation(cooc) for cooc in coocs]
        contr = [contrast(cooc) for cooc in coocs]
        diss = [dissimilarity(cooc) for cooc in coocs]
        en = [energy(cooc) for cooc in coocs]
        entr = [entropy(cooc) for cooc in coocs]
        hom = [homogeneity(cooc) for cooc in coocs]
        inv_diff = [inverse_difference(cooc) for cooc in coocs]
        max_prob = [maximum_probability(cooc) for cooc in coocs]

        # Transposing the tuples to have descriptors per co-occurrence matrix
        descs = list(zip(*[auto_corr, contr, diss, en, entr, hom, inv_diff, max_prob]))
        descs = np.concatenate((descs[0], descs[1]), axis=None)
        descriptors.append(descs)
    return descriptors


def order_by_similarity():
    pass


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
    morphed_images = list()
    if F == 1:
        morphed_images = opening(bin_images)
    elif F == 2:
        morphed_images = closing(bin_images)

    masks = masks_from(gray_images, morphed_images)

    # part 2
    cooc_matrixes = cooccurrence_prob_matrixes(masks)

    sep_descriptors = compute_haralick_descriptors(cooc_matrixes)


if __name__ == "__main__":
    run()
