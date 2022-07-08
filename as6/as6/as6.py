# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 6 - Color Image Processing and Segmentation
from typing import List, Tuple

import imageio
import numpy as np
import numpy.typing as npt
import random


def rmse(ref: npt.NDArray[np.float32], gen: npt.NDArray[np.float32]) -> float:
    """Root Mean Squared Error calculation
    Arguments:
        ref: The reference image
        gen: The generated image

    Returns:
        The Root Mean Squared Error between both images
    """
    m, n = gen.shape
    subtracted_image = gen - ref
    squared_image = np.square(subtracted_image)
    mean_image = squared_image / (n * m)
    err = np.sum(mean_image)

    return np.sqrt(err)


def root_mean_sq_err(
    ref_image: npt.NDArray[np.float32], gen_image: npt.NDArray[np.float32]
) -> float:
    """Root Mean Squared Error wrapper function.
    This function separates the rmse calculation between
    grayscale and RGB images

    Arguments:
        ref: The reference image
        gen: The generated image

    Returns:
        The Root Mean Squared Error between both images
    """
    _, _, c = gen_image.shape
    if c == 1:
        return rmse(ref_image, np.squeeze(gen_image))
    else:
        err = [rmse(ref_image[:, :, i], gen_image[:, :, i]) for i in range(c)]
        return np.mean(err)


def distance(
    features: npt.NDArray[np.float32], centroid: npt.NDArray[np.float32]
) -> float:
    """Euclidean distance between two arrays

    Arguments:
        features: Image features array
        centroid: Current centroid in analysis by the algorithm

    Returns:
        The Euclidean distance between both arrays
    """
    return np.sqrt(np.sum(np.power(features - centroid, 2), axis=1))


def init_centroids(
    features: npt.NDArray[np.float32], seed: int, M: int, N: int, k: int
) -> npt.NDArray[np.float32]:
    """Initialisation of centroids for k_means algorithm

    Arguments:
        features: Image features array
        seed: The random seed to be used
        M: The number of rows in the image
        N: The number of columns in the image
        k: The number of clusters for the k_means algorithm

    Returns:
        The initial centroids acquired from the features array
    """
    random.seed(seed)
    ids = np.sort(random.sample(range(0, M * N), k))
    centroids = np.array(features[ids])
    return centroids


def k_means(
    features: npt.NDArray[np.float32],
    k: int,
    n: int,
    M: int,
    N: int,
    seed: int,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """K-Means algorith calculation function

    Arguments:
        features: Image features array
        k: The number of clusters for the k_means algorithm
        n: The number of iterations of the k_means algorithm
        M: The number of rows in the image
        N: The number of columns in the image
        seed: The random seed to be used

    Returns:
        A tuple containing the final centroids acheived through
    the algorithm, and the corresponding clusters for each image
    feature.
    """
    centroids = init_centroids(features, seed, M, N, k)
    clusters = np.zeros(features.shape[0], dtype=np.float32)

    for _ in range(n):
        distances = np.zeros((features.shape[0], k), dtype=np.float32)

        for cluster in range(k):
            centroid = np.full(features.shape, centroids[cluster])
            dist = distance(features, centroid)
            distances[:, cluster] = dist

        clusters = np.argmin(distances, axis=1)

        for cluster in range(k):
            centroids[cluster] = np.mean(
                features[np.where(clusters == cluster)], axis=0
            )

    return centroids, clusters


def create_XY_features(M: int, N: int) -> npt.NDArray[np.float32]:
    """Helper function to create the positional features for
    attribute_type cases 2 and 4

    Arguments:
        M: The number of rows in the image
        N: The number of columns in the image

    Returns:
        An array with all positional features created
    """
    X = np.tile(np.reshape(np.arange(M), (M, 1)), (1, N))
    Y = np.tile(np.reshape(np.arange(N), (N, 1)), (M, 1))
    X = np.reshape(X, (M * N, 1))
    Y = np.reshape(Y, (M * N, 1))
    XY = np.concatenate((X, Y), axis=1)
    return XY


def lum_reshape(
    image: npt.NDArray[np.float32], M: int, N: int
) -> npt.NDArray[np.float32]:
    """Helper method to reshape and transform the image to grayscale
    based on luminance

    Arguments:
        image: The image to be converted
        M: The number of rows in the image
        N: The number of columns in the image

    Returns:
        The converted image in grayscale and reshaped to a single column.
    """
    return np.reshape(
        (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2]),
        (M * N, 1),
    )


def _normalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Image normalisation function

    Arguments:
        image: The image to be normalised

    Returns:
        The normalised image between 0 and 255
    """
    image = (image - image.min()) / (image.max() - image.min())
    image *= 255.0
    return image


def normalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Image normalisation wrapper function to separate between normalisation
    of grayscale and RGB images

    Arguments:
        image: The image to be normalised

    Returns:
        The normalised image between 0 and 255
    """
    _, _, channels = image.shape
    if channels == 3:
        for channel in range(channels):
            image[:, :, channel] = _normalize_image(image[:, :, channel])
    else:
        image = _normalize_image(image)

    return image


def rgb_image_from(
    centroids: List[npt.NDArray[np.float32]],
    clusters: List[int],
    k: int,
    img_shape: Tuple[int, int, int],
):
    """RGB image creation function from calculated centroids

    Arguments:
        centroids: The calculated centroids from k_means algorithm
        clusters: The clusters corresponding to each image feature
        k: The number of clusters
        img_shape: The expected shape of the resulting image

    Returns:
        The newly created image with pixel values from the
    corresponding centroids
    """
    out_image = centroids[clusters % k][:, :3]
    out_image = np.reshape(out_image, img_shape)
    out_image = normalize_image(out_image)
    return out_image


def gray_image_from(
    centroids: List[npt.NDArray[np.float32]],
    clusters: List[int],
    k: int,
    img_shape: Tuple,
):
    """Grayscale image creation function from calculated centroids

    Arguments:
        centroids: The calculated centroids from k_means algorithm
        clusters: The clusters corresponding to each image feature
        k: The number of clusters
        img_shape: The expected shape of the resulting image

    Returns:
        The newly created image with pixel values from the
    corresponding centroids
    """
    out_image = centroids[clusters % k][:, 0]
    out_image = np.reshape(out_image, img_shape)
    out_image = normalize_image(out_image)
    return out_image


def main():
    image_filename = str(input().strip())
    ref_image_filename = str(input().strip())
    attributes_type = int(input().strip())
    k = int(input().strip())
    n = int(input().strip())
    seed = int(input().strip())

    image = imageio.imread(image_filename).astype(np.float32)
    ref_image = imageio.imread(ref_image_filename).astype(np.float32)

    M, N, _ = image.shape

    # Selecting flow based on attributes_type
    if attributes_type == 1:
        features = np.reshape(image, (M * N, 3))
        centroids, clusters = k_means(features, k, n, M, N, seed)
        out_image = rgb_image_from(centroids, clusters, k, (M, N, 3))
    elif attributes_type == 2:
        XY = create_XY_features(M, N)
        features = np.reshape(image, (M * N, 3))
        features = np.concatenate((features, XY), axis=1)

        centroids, clusters = k_means(features, k, n, M, N, seed)
        out_image = rgb_image_from(centroids, clusters, k, (M, N, 3))
    elif attributes_type == 3:
        features = lum_reshape(image, M, N)
        centroids, clusters = k_means(features, k, n, M, N, seed)
        out_image = gray_image_from(centroids, clusters, k, (M, N, 1))
    elif attributes_type == 4:
        features = lum_reshape(image, M, N)
        features = np.concatenate((features, create_XY_features(M, N)), axis=1)

        centroids, clusters = k_means(features, k, n, M, N, seed)
        out_image = gray_image_from(centroids, clusters, k, (M, N, 1))
    else:
        # Bogus entry
        exit(1)

    rmse = root_mean_sq_err(ref_image, out_image)
    print(rmse)


if __name__ == "__main__":
    main()
