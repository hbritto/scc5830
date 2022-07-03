# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 5 - Morphology and Image Description
import os

import imageio
import numpy as np
import numpy.typing as npt
from typing import List, Tuple


def structure_output(
    query: Tuple[str, npt.NDArray[np.float32]],
    results: List[Tuple[str, npt.NDArray[np.float32]]],
):
    """Function for structuring the script's output, matching the expected for run.codes

    Arguments:
        query: Tuple containing the query image name and corresponding descriptors
        results: list of tuples containing image name and corresponding descriptors for all images input
    """
    output = f"Query: {query[0]}\n"
    output += "Ranking:\n"
    for i, result in enumerate(results):
        output += f"({i}) {result[0]}\n"  # | Similarity: {result[1]}\n"
    print(output[:-1])


def load_images(image_names: List[str]) -> npt.NDArray[np.float32]:
    """Helper function for loading images given a path

    Arguments:
        image_names: list of strings containing the paths of images to be loaded.

    Returns: list of ndarrays containing the images loaded in float32 format.
    """
    images = [imageio.imread(image_name) for image_name in image_names]
    images = list(map(lambda img: img.astype(np.float32), images))
    return images


def convert_gray_luminance(
    images: List[npt.NDArray[np.float32]],
) -> List[npt.NDArray[np.uint8]]:
    """Image conversion from RGB to grayscale using the Luminance method

    Arguments:
        images: list of images in float32 format

    Returns:
        The images converted to grayscale in uint8 format
    """
    lgrays = []
    for image in images:
        lgray = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
        lgrays.append(lgray.astype(np.uint8))
    return lgrays


def binarize_images(
    images: List[npt.NDArray[np.uint8]], T: int
) -> List[npt.NDArray[np.uint8]]:
    """Binarization of the images according to a threshold

    Arguments:
        images: list of grayscale images in uint8 format

    Returns:
        The images binarized
    """
    bin_imgs = []
    for image in images:
        bin_img = np.copy(image)
        bin_img[image < T] = 0
        bin_img[image >= T] = 1
        bin_imgs.append(bin_img)
    return bin_imgs


def erode_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Binary image erosion method based on a 3x3 square

    Arguments:
        image: the image to be eroded

    Returns:
        The eroded image
    """
    eroded = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(1, rows):
        for j in range(1, cols):
            min_val = np.min(image[i - 1 : i + 2, j - 1 : j + 2])
            eroded[i, j] = min_val
    return eroded


def dilate_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Binary image dilation method based on a 3x3 square

    Arguments:
        image: the image to be dilated

    Returns:
        The dilated image
    """
    dilated = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(1, rows):
        for j in range(1, cols):
            max_val = np.max(image[i - 1 : i + 2, j - 1 : j + 2])
            dilated[i, j] = max_val
    return dilated


def erode_images(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
    """Helper mapping function for eroding a list of images

    Arguments:
        images: List of images to be eroded

    Returns:
        List of eroded images
    """
    eroded = list(map(erode_image, images))
    return eroded


def dilate_images(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
    """Helper mapping function for dilating a list of images

    Arguments:
        images: List of images to be dilated

    Returns:
        List of dilated images
    """
    dilated = list(map(dilate_image, images))
    return dilated


def closing(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
    """Helper function for applying morphological closing to a list of images

    Arguments:
        images: List of images for closing to be applied

    Returns:
        List of images after closing application
    """
    dilated = dilate_images(images)
    closed = erode_images(dilated)
    return closed


def opening(images: List[npt.NDArray[np.uint8]]) -> List[npt.NDArray[np.uint8]]:
    """Helper function for applying morphological opening to a list of images

    Arguments:
        images: List of images for opening to be applied

    Returns:
        List of images after opening application
    """
    eroded = erode_images(images)
    opened = dilate_images(eroded)
    return opened


def masks_from(
    gray_images: List[npt.NDArray[np.uint8]],
    morphed_images: List[npt.NDArray[np.uint8]],
) -> List[Tuple[npt.NDArray[np.uint8]]]:
    """Calculation of two masks for each image based on morphological alterations applied

    Arguments:
        gray_images: list of images in grayscale
        morphed_images: list of images after morphological alterations.

    Returns:
        List of tuples with two masks for each image received
    """
    masks = []
    for gray, morphed in zip(gray_images, morphed_images):
        mask1 = np.zeros_like(gray)
        mask2 = np.zeros_like(gray)
        mask1[morphed == 0] = gray[morphed == 0]
        mask2[morphed == 1] = gray[morphed == 1]
        masks.append((mask1, mask2))
    return masks


def cooccurrence_prob_matrix(
    mask: npt.NDArray[np.uint8], Q: List[int]
) -> npt.NDArray[np.float32]:
    """Calculation of the co-occurrence matrix for each mask based on the neighbour Q,
    and further converting it to a probability matrix

    Arguments:
        mask: an image mask
        Q: the neibouring pixel to be considered

    Returns:
        The probability matrix for the supplied mask
    """
    row, col = mask.shape
    cooc_side = np.max(mask) + 1
    cooc = np.zeros((cooc_side, cooc_side))
    for x in range(1, row - 1):
        for y in range(1, col - 1):
            ref = mask[x, y]
            neigh = mask[x + Q[0], y + Q[1]]
            cooc[ref, neigh] += 1
    prob_matrix = cooc / np.sum(cooc)
    return prob_matrix


def cooccurrence_prob_matrixes(
    masks: List[Tuple[npt.NDArray[np.uint8]]], Q: List[int]
) -> List[List[npt.NDArray[np.float32]]]:
    """Helper method to calculate the probability matrix for every mask supplied

    Arguments:
        masks: list of pairs of masks per originating image
        Q: Neighbour point coordinate to be used for calculation

    Returns:
        All probability matrixes calculated
    """
    matrixes = []
    for mm in masks:
        coocs = [cooccurrence_prob_matrix(m, Q) for m in mm]
        matrixes.append(coocs)
    return matrixes


def vectors_for(prob_matrix: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.int64]]:
    """Helper method for retrieving I and J vectors for a probability matrix

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The vectors I and J to be utilised in descriptors calculations
    """
    M, N = prob_matrix.shape
    I, J = np.ogrid[0:M, 0:N]
    return I, J


def auto_correlation(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Auto correlation Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The auto correlation Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return (prob_matrix * (I * J)).sum()


def contrast(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Contrast Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The contrast Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return (np.square(I - J) * prob_matrix).sum()


def dissimilarity(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Dissimilarity Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The dissimilarity Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return (np.abs(I - J) * prob_matrix).sum()


def energy(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Energy Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The energy Haralick descriptor
    """
    return np.square(prob_matrix).sum()


def entropy(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Entropy Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The entropy Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return -(prob_matrix[prob_matrix > 0] * np.log(prob_matrix[prob_matrix > 0])).sum()


def homogeneity(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Homogeneity Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The homogeneity Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return (prob_matrix / (1 + np.square(I - J))).sum()


def inverse_difference(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Inverse difference Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The inverse difference Haralick descriptor
    """
    I, J = vectors_for(prob_matrix)
    return (prob_matrix / (1 + np.abs(I - J))).sum()


def maximum_probability(prob_matrix: npt.NDArray[np.float32]) -> np.float32:
    """Maximum probability Haralick descriptor calculation

    Arguments:
        prob_matrix: the probability matrix

    Returns:
        The maximum probability Haralick descriptor
    """
    return np.max(prob_matrix)


def compute_haralick_descriptors(
    prob_matrixes: List[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
) -> List[npt.NDArray[np.float32]]:
    """Function for calculating and organising Haralick descriptors
    for every probability matrix received

    Arguments:
        prob_matrixes: list of probability matrixes

    Returns:
        List of all Haralick descriptors concatenated for every two probability matrixes
        which correspond to one image
    """
    descriptors = []
    for prob_matrix in prob_matrixes:
        auto_corr = [auto_correlation(prob_matrix) for prob_matrix in prob_matrix]
        contr = [contrast(prob_matrix) for prob_matrix in prob_matrix]
        diss = [dissimilarity(prob_matrix) for prob_matrix in prob_matrix]
        en = [energy(prob_matrix) for prob_matrix in prob_matrix]
        entr = [entropy(prob_matrix) for prob_matrix in prob_matrix]
        hom = [homogeneity(prob_matrix) for prob_matrix in prob_matrix]
        inv_diff = [inverse_difference(prob_matrix) for prob_matrix in prob_matrix]
        max_prob = [maximum_probability(prob_matrix) for prob_matrix in prob_matrix]

        # Transposing the tuples to have descriptors per probability matrix
        mask1_desc, mask2_desk = list(
            zip(*[auto_corr, contr, diss, en, entr, hom, inv_diff, max_prob])
        )
        all_descs = np.concatenate((mask1_desc, mask2_desk), axis=None)
        descriptors.append(all_descs)
    return descriptors


def calc_distance(
    query: npt.NDArray[np.float32], reference: npt.NDArray[np.float32]
) -> np.float32:
    """Euclidean distance helper function

    Arguments:
        query: Haralick descriptors of the query image
        reference: Haralick descriptors of the reference image

    Returns:
        The euclidean distance between both descriptors
    """
    return np.linalg.norm(query - reference)


def order_by_similarity(
    query: Tuple[str, npt.NDArray[np.float32]],
    descriptors: List[Tuple[str, npt.NDArray[np.float32]]],
):
    """Ordering the descriptors for all images by similarity from the query

    Arguments:
        query: the query image
        descriptors: Haralick descriptors for all images loaded

    Returns:
        List of images ordered by similarity from query
    """
    distances = [
        (ref_name, calc_distance(query[1], ref_desc))
        for ref_name, ref_desc in descriptors
    ]
    maxdist = max([d[1] for d in distances])
    distances = [(name, (maxdist - dist) / maxdist) for name, dist in distances]
    similarities = [(name, dist) for name, dist in distances]
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities


def run():
    query_index = int(input().strip())
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

    # Start of part 2
    cooc_matrixes = cooccurrence_prob_matrixes(masks, Q)

    all_descriptors = compute_haralick_descriptors(cooc_matrixes)

    descriptors_per_images = list(zip(image_names, all_descriptors))
    query_image = (image_names[query_index], all_descriptors[query_index])

    results = order_by_similarity(query_image, descriptors_per_images)

    structure_output(query_image, results)


if __name__ == "__main__":
    run()
