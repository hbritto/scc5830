# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 4 - Image Restoration

import imageio
import numpy as np

def root_mean_sq_err(ref_image: np.ndarray, gen_image: np.ndarray) -> float:
    """RMSE calculation between a reference and a generated image

    Arguments:
        ref_image: The reference image
        gen_image: The generated image

    Returns:
        The Root Mean Square Error between the images
    """
    n, m = gen_image.shape
    subtracted_image = gen_image.astype(float) - ref_image.astype(float)
    squared_image = np.square(subtracted_image)
    mean_image = squared_image / (n * m)
    err = np.sum(mean_image)

    return np.sqrt(err)


def fft(image: np.ndarray) -> np.ndarray:
    """Applies the Fast Fourier Transform to an image and returns its result.

    Arguments:
        image: The image in spacial domain.

    Returns:
        The image in Frequency domain.
    """
    freq_image = np.fft.fftn(image)
    return freq_image


def ifft(image: np.ndarray) -> np.ndarray:
    """Applies the Inverse Fast Fourier Transform to an image and returns its result.

    Arguments:
        image: The image in frequency domain and shifted.

    Returns:
        The image in spacial domain.
    """
    spac_image = np.fft.ifftn(image)
    return spac_image.real


def conv_point_2d(
    big: np.ndarray, small: np.ndarray, x: int, y: int, a: int, b: int
) -> int:
    """Single iteration of a 2D convolution.

    Arguments:
        big: Bigger matrix
        small: smaller matrix
        x: central column in the convolution
        y: central row in the convolution
        a: half the vertical size of the convolution
        b: half the horizontal size of the convolution

    Returns:
        The point result of the convolution
    """
    sub_big = big[x - a : x + a + 1, y - b : y + b + 1]
    res = np.sum(np.multiply(sub_big, small))
    return int(res)


def convolve_2d(big: np.ndarray, small: np.ndarray) -> np.ndarray:
    """Convolution of 2D matrixes.

    Arguments:
        big: The bigger matrix in the convolution.
        small: The smaller matrix in the convolution.

    Returns:
        The convolution matrix result.
    """
    rows, cols = big.shape
    n, m = small.shape
    a = int(rows // 2 - n // 2)
    b = int(cols // 2 - m // 2)
    small_flipped = np.flip(np.flip(small, 0), 1)
    big_padded = np.pad(big, (a, b), "edge")
    result = np.zeros(big.shape)

    for i in range(a, rows):
        for j in range(b, cols):
            result[i - a, j - b] = conv_point_2d(big_padded, small_flipped, i, j, a, b)

    return result


def gaussian_filter(k: int, sigma: float) -> np.ndarray:
    """Function that creates a gaussian blur filter.
    This function has been obtained from the study material.

    Arguments:
        k: The size of the kernel
        sigma: The std. deviation of the Gaussian curve.

    Returns:
        A gaussian filter matrix, also called Point Spread Function in this context.
    """
    range = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(range, range)
    fil = np.exp(-(1 / 2) * (np.square(x) + np.square(y)) / np.square(sigma))
    norm_filter = fil / np.sum(fil)
    return norm_filter


def apply_gaussian_filter(
    image: np.ndarray, k: int, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Changes the image received by applying a Gaussian blur filter to it.

    Arguments:
        image: The original image in space domain.
        k: Size of the side of the square matrix filter.
        sigma: Standard deviation for the Gaussian filter generation.

    Returns:
        A tuple containing a new image with Gaussian filter applied and the filter itself.
    """

    gaussian_f = gaussian_filter(k, sigma)
    pad_size = int(image.shape[0] // 2 - gaussian_f.shape[0] // 2)
    padded_filter = np.pad(gaussian_f, (pad_size, pad_size), "constant")
    freq_filter = fft(padded_filter)
    freq_image = fft(image)
    blurred_image = np.multiply(freq_image, freq_filter)
    return np.fft.fftshift(ifft(blurred_image)), gaussian_f


def clsq(image: np.ndarray, psf: np.ndarray, gamma: float) -> np.ndarray:
    """Constrained Least Squares method application.

    Arguments:
        image: The blurred image in frequency domain (G).
        psf: The point-spread-function representing the blur in the image (H).
        gamma: The multiplication parameter to the laplacian operator.

    Returns:
        The restored image after CLSQ application.
    """
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    l_pad = int(image.shape[0] // 2 - laplacian.shape[0] // 2)
    psf_pad = int(image.shape[0] // 2 - psf.shape[0] // 2)
    padded_lap = np.pad(laplacian, (l_pad, l_pad), "constant")
    padded_psf = np.pad(psf, (psf_pad, psf_pad), "constant")
    G = fft(image)
    P = fft(padded_lap)
    H = fft(padded_psf)

    restored_image = (
        H.conj() / (np.square(np.abs(H)) + (gamma * np.square(np.abs(P))))
    ) * G
    restored_space_image = np.fft.fftshift(ifft(restored_image))
    restored_space_image = np.clip(restored_space_image, 0, 255)
    return restored_space_image.astype(np.uint8)


def get_motion_psf(
    dim_x: int, dim_y: int, degree_angle: float, num_pixel_dist: int = 20
) -> np.ndarray:
    """PSF creation for a motion blur simulation base on a rotation angle.
    This function was obtained from the study material

    Arguments:
        dim_x: The width of the image.
        dim_y: The height of the image.
        degree_angle: The angle of the motion blur. Should be in degrees. [0, 360)
        num_pixel_dist: The distance of the motion blur. [0, \infinity).
            The distance is measured in pixels.
            Greater will be more blurry.

    Returns:
    --------
        np.ndarray
            The point-spread array associated with the motion blur.

    """
    psf = np.zeros((dim_x, dim_y))
    center = np.array([dim_x - 1, dim_y - 1]) // 2
    radians = degree_angle / 180 * np.pi
    phase = np.array([np.cos(radians), np.sin(radians)])
    for i in range(num_pixel_dist):
        offset_x = int(center[0] - np.round_(i * phase[0]))
        offset_y = int(center[1] - np.round_(i * phase[1]))
        psf[offset_x, offset_y] = 1
    psf /= psf.sum()

    return psf


def richardson_lucy_filter(
    image: np.ndarray, psf: np.ndarray, max_iter: int = 50
) -> np.ndarray:
    """RF Filter application function.

    Arguments:
        image: The degraded image.
        psf: The supposed Point Spread Function.
        max_iter: Maximum number of iterations.

    Returns:
        The restored image.
    """
    O_k = np.ones(image.shape)
    O_next = np.copy(O_k)
    epsilon = 0.001
    for _ in range(max_iter):
        denom = convolve_2d(O_k, psf)
        frac = image / (denom + epsilon)
        partial = convolve_2d(frac, np.flip(psf))
        O_next = O_k * partial
        O_k = O_next

    restored = ifft(O_k)
    restored = np.clip(restored, 0, 255)
    return restored.astype(np.uint8)


def main():
    in_image_filename = str(input().rstrip())
    method = int(input().strip())
    if method == 1:
        k = int(input().strip())
        sigma = float(input().strip())
        gamma = float(input().strip())
        base_image = imageio.imread(in_image_filename)
        degraded, psf = apply_gaussian_filter(base_image, k, sigma)
        restored = clsq(degraded, psf, gamma)
    else:
        angle = int(input().strip())
        steps = int(input().strip())
        base_image = imageio.imread(in_image_filename)
        restored = richardson_lucy_filter(
            base_image,
            get_motion_psf(
                base_image.shape[0], base_image.shape[1], degree_angle=angle
            ),
            steps,
        )

    rmse = root_mean_sq_err(base_image, restored)
    print(f"{rmse:.4f}")


if __name__ == "__main__":
    main()
