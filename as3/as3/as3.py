# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 3 - Filtering in Spatial and Frequency Domain

import numpy as np
import imageio


def dft2d(image):
    """Wrapper for the numpy implementation of Fast Fourier Transform for 2-dimensional arrays

    Arguments:
        image: Space domain image

    Returns:
        The image in frequency domain
    """
    return np.fft.fft2(image)


def idft2d(freq_image):
    """Wrapper for the numpy implementation of Inverse Fast Fourier Transform for 2-dimensional arrays

    Arguments:
        freq_image: Frequency domain image

    Returns:
        The image in space domain
    """
    return np.fft.ifft2(freq_image)


def prepare_filter(filter_image):
    """Normalisation of the filter to be applied

    Arguments:
        filter_image: Image representing the filter to be applied

    Returns:
        The filter normalised to 0's and 1's
    """
    return filter_image // 255


def frequency_domain_filter(image, filter_image):
    """Application of the filter to in image in frequency domain

    For the filter to be correctly applied it is necessary that the image in
    frequency domain has the lower frequencies located at the center and the
    higher at the edges of the array. That is achieved through the 'shift'
    method.
    After the filter is applied, the image has to be 'unshifted' so that, when
    returned to space domain, it is correct, as expected.

    Arguments:
        image: The image in frequency domain
        filter_image: The filter to be applied

    Returns:
        The filtered image in frequency domain
    """
    shifted = np.fft.fftshift(image)
    filtered_shifted = np.multiply(shifted, filter_image)
    filtered = np.fft.ifftshift(filtered_shifted)
    return filtered


def normalise_image(image):
    """Normalisation of the output image between 0 and 255

    Arguments:
        image: Space domain image

    Returns:
        The image normalised between 0 and 255
    """
    imax = np.max(image)
    imin = np.min(image)
    image_norm = (image - imin) / (imax - imin)
    image_norm = (image_norm * 255).astype(np.uint8)
    return image_norm


def root_mean_sq_err(ref_image, gen_image):
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


def read_from_stdin():
    """Wrapper method for reading from stdin and removing trailing characters

    Returns:
        The stripped input from stdin
    """
    return str(input()).strip()


def main():
    # Reading program arguments
    in_image_filename = read_from_stdin()
    filter_filename = read_from_stdin()
    ref_image_filename = read_from_stdin()
    in_image = imageio.imread(in_image_filename)
    filter_image = imageio.imread(filter_filename)
    ref_image = imageio.imread(ref_image_filename)

    # Converting the image to frequency domain
    freq_image = dft2d(in_image)

    # Normalising the filter to be applied
    norm_filter = prepare_filter(filter_image)

    # Applying the filter to the image and converting it back to space domain
    filtered_image = frequency_domain_filter(freq_image, norm_filter)
    final_image = idft2d(filtered_image)

    # Normalising generated image and calculating its RMSE against the reference
    normalised_final_image = normalise_image(final_image.real)
    rmse = root_mean_sq_err(ref_image, normalised_final_image)
    print(f"{rmse:.4f}")


if __name__ == "__main__":
    main()
