from collections import defaultdict
from typing import Dict, Tuple

import imageio
import numpy as np
import numpy.typing as npt
import random
import os


# Calculate error
def root_mean_sq_err(ref_image, gen_image):
    m, n = gen_image.shape
    subtracted_image = gen_image.astype(float) - ref_image.astype(float)
    squared_image = np.square(subtracted_image)
    mean_image = squared_image / (n * m)
    err = np.sum(mean_image)

    return np.sqrt(err)


def distance(
    query: npt.NDArray[np.float32], reference: npt.NDArray[np.float32]
) -> float:
    return np.linalg.norm(query - reference)


def find_closest_cluster(
    feature: npt.NDArray[np.float32], centroids: npt.NDArray[np.float32]
) -> Tuple[int, npt.NDArray[np.float32]]:
    centr_ranks = [
        (idx, distance(feature, centroid)) for idx, centroid in enumerate(centroids)
    ]
    centr_ranks.sort(key=lambda x: x[1])
    return centr_ranks[0]


def k_means(
    image: npt.NDArray[np.float32],
    features: npt.NDArray[np.float32],
    k: int,
    n: int,
    M: int,
    N: int,
    seed: int,
) -> Dict[int, npt.NDArray[np.float32]]:
    random.seed(seed)
    ids = np.sort(random.sample(range(0, M * N), k))
    centroids = features[ids]
    clustered = defaultdict(list)
    loc_centroid = dict()
    for _ in range(n):
        clustered.clear()
        for index, feature in enumerate(features):
            closest_idx, _ = find_closest_cluster(feature, centroids)
            clustered[closest_idx].append(feature)
            loc_centroid[index] = centroids[closest_idx]
        for idx, feats in clustered.items():
            centroid = np.mean(feats)
            centroids[idx] = centroid
    return loc_centroid


def create_XY_features(M, N):
    X = np.tile(np.reshape(np.arange(M), (M, 1)), (1, N))
    Y = np.tile(np.reshape(np.arange(N), (N, 1)), (M, 1))
    X = np.reshape(X, (M * N, 1))
    Y = np.reshape(Y, (M * N, 1))
    XY = np.concatenate(X, Y, axis=1)
    return XY


def lum_reshape(image, M, N):
    return np.reshape(
        (0.299 * image[:, :, 0]) + (0.587 * image[:, :, 1]) + (0.114 * image[:, :, 2]),
        (M * N, 1),
    )


def image_from(clusters, feat_shape, img_shape):
    out_image = np.zeros(feat_shape, dtype=np.float32)
    for index, centroid in clusters.items():
        out_image[index] = centroid
    out_image = np.reshape(out_image, img_shape)
    return out_image

def main():
    # BASE_PATH = '../tests/resources/TestCases-InputImages'
    image_filename = str(input().strip())
    ref_image_filename = str(input().strip())
    attributes_type = int(input().strip())
    k = int(input().strip())
    n = int(input().strip())
    seed = int(input().strip())

    image = imageio.imread(image_filename).astype(np.float32)
    ref_image = imageio.imread(ref_image_filename).astype(np.float32)

    M, N, _ = image.shape
    rgb = 1

    if attributes_type == 1:
        features = np.reshape(image, (M*N, 3))
    elif attributes_type == 2:
        XY = create_XY_features(M, N)
        features = np.reshape(image, (M*N, 3))
        features = np.concatenate(features, XY, axis=1)
    elif attributes_type == 3:
        features = lum_reshape(image, M, N)
        rgb = 0
    elif attributes_type == 4:
        features = lum_reshape(image, M, N)
        features = np.concatenate(features, create_XY_features(M, N), axis=1)
        rgb = 0
    else:
        exit(1)

    clusters = k_means(image, features, k, n, M, N, seed)
    out_image = image_from(clusters, features.shape, (M, N, 3) if rgb else (M, N, 1))
    rmse = root_mean_sq_err(ref_image, out_image)
    print(rmse)

if __name__ == '__main__':
    main()