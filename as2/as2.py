# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 2 - Image Enhancement and Filtering
import imageio
import numpy as np
    
def limiarization(image, t0):
    """Function to limiarize a given image
    
    Arguments:
        image: The image to be limiarized
        t0: the initial threshold

    Returns:
        The optimal threshold T
    """

    def calc_threshold(image, t0):
        """Function to calculate the optimal threshold T
        based on an initial value T0
        
        Arguments:
            image: The image to binarize
            t0: the initial threshold

        Returns:
            The optimal threshold T
        """
        # Saving the initial threshold
        t = t0
        # Initialising the 'last iteration threshold' to guarantee at least one calculation
        t0 = 1000000
        while abs(t - t0) >= 0.5:
            # Separating the image in groups and calculating the mean of each group
            group_1 = image[image <= t]
            group_2 = image[image > t]
            avg_1 = np.mean(group_1)
            avg_2 = np.mean(group_2)

            t0 = t
            # Updating the threshold
            t = (avg_1 + avg_2)/2
        
        return t

    thresh = calc_threshold(image, t0)
    lim_image = np.zeros(image.shape, dtype=np.uint8)

    # Limiarizing the image
    lim_image[image <= thresh] = 0
    lim_image[image > thresh] = 1

    return lim_image

def filter_1d(image, n, weights):
    """Function to filter a given image vetorizing and calculating
    in 1 dimension
    
    Arguments:
        image: The image to filter
        n: the size of the 1D filter
        weights: the values of the 1D filter

    Returns:
        The filtered image
    """

    def conv_point_1d(image, weights, x, a):
        """Function to calculate the convolution of a single point in an image
        based on a 1D filter
        
        Arguments:
            image: The original image
            weights: the 1D filter
            x: the point in the resulting image being calculated
            a: the point coordinate offset

        Returns:
            The resulting point value
        """
        # Slicing the intended section of the image and applying the filter
        sub_arr = image[x-a:x+a+1]
        res = np.sum(np.multiply(sub_arr, weights))
        return int(res)

    # Converting the image to 1D and calculating the point offset
    image_1d = image.flatten()
    length_1d = image_1d.shape[0]
    w_offset = int((n - 1)/2)
    # Correcting the vectors to use the cross-correlation function
    # and get the correct convolution results
    padded_1d = np.pad(image_1d, w_offset, 'wrap')
    weights_flipped = np.flip(weights)
    result_1d = np.zeros(length_1d)
    
    # Calculating the resulting image
    for i in range(w_offset, length_1d):
        result_1d[i-w_offset] = conv_point_1d(padded_1d, weights_flipped, i, w_offset)
    
    # Transforming the 1D image back to 2D
    result = result_1d.reshape(image.shape).astype(np.uint8)
    return result


def filter_2d(image, n, weights):
    """Function to filter a given image by applying convolution of a 2D filter
    
    Arguments:
        image: The image to filter
        n: the size of the 2D filter
        weights: the values of the 2D filter

    Returns:
        The filtered image
    """

    def conv_point_2d(image, weights, x, y, a, b):
        """Function to calculate the convolution of a single point in an image
        based on a 2D filter
        
        Arguments:
            image: The original image
            weights: the 2D filter
            x: the point in the resulting image being calculated
            y: the point in the resulting image being calculated
            a: the point x coordinate offset
            b: the point y coordinate offset

        Returns:
            The resulting point value
        """
        # Slicing the intended section of the image and applying the filter
        sub_image = image[x-a:x+a+1, y-b:y+b+1]
        res = np.sum(np.multiply(sub_image, weights))
        return int(res)

    # Initialising auxiliary variables and calculating coordinate offsets
    rows, cols = image.shape
    n, m = weights.shape
    a = (n - 1)//2
    b = (m - 1)//2
    # Correcting the matrixes to use the cross-correlation function
    # and get the correct convolution results
    weights_flipped = np.flip(weights, (0, 1))
    padded_image = np.pad(image, (a, b), 'edge')
    result = np.zeros(image.shape)

    # Calculating the resulting image
    for i in range(a, rows):
        for j in range(b, cols):
            result[i-a, j-b] = conv_point_2d(padded_image, weights_flipped, i, j, a, b)
    
    return result


def median_filter(image, n):
    """Function to apply a median filter to a given image
        
    Arguments:
        image: The original image
        n: the size of the median filter

    Returns:
        The filtered image
    """

    def median_point(image, offset, x, y):
        """Function to calculate median of a section of a given image
        
        Arguments:
            image: The original image
            offset: the size of the subsection of the image to be considered
            x: the point in the middle of the subsection of the image
            y: the point in the middle of the subsection of the image

        Returns:
            The median of the section of the image
        """

        # Slicing the image, flattening and sorting it
        sub_image = image[x-offset:x+offset+1, y-offset:y+offset+1]
        sub_image = sub_image.flatten()
        sub_image = sorted(sub_image)
        size_sub_image = len(sub_image)

        # Manually calculating the median to avoid timeouts with the numpy.median function
        if size_sub_image % 2 == 0:
            median = (sub_image[size_sub_image // 2] + sub_image[(size_sub_image // 2) + 1]) // 2
        else:
            median = sub_image[size_sub_image // 2]
        return median

    # Initialising auxiliary variables and calculating the offset of the filter
    rows, cols = image.shape
    offset = (n - 1)//2
    padded_image = np.pad(image, offset, 'constant')
    result = np.zeros(image.shape)

    # Applying the median filter to the image
    for i in range(offset, rows):
        for j in range(offset, cols):
            result[i-offset, j-offset] = median_point(padded_image, offset, i ,j)
    
    return result

# Calculate error
def calculate_root_mean_sq_err(ref_image, gen_image):
    """Function to calculate the RMSE of an image given a reference
    
    Arguments:
        ref_image: The reference image
        gen_image: The image to be compared

    Returns:
        The RMSE between the two images
    """
    n, m = gen_image.shape
    subtracted_image = gen_image.astype(float) - ref_image.astype(float)
    squared_image = np.square(subtracted_image)
    mean_image = squared_image/(n*m)
    err = np.sum(mean_image)
    
    return np.sqrt(err)

if __name__ == '__main__':

    # read program arguments
    input_image = input().strip()
    method = int(input().strip())
    if method == 1:
        t0 = int(input().strip())
    elif method == 2:
        n = int(input().strip())
        weights = np.array(input().strip().split(' '), dtype=np.float64)
    elif method == 3:
        n = int(input().strip())
        weights = np.array([input().strip().split(' ') for i in range(n)], dtype=np.float64)
        t0 = int(input().strip())
    elif method == 4:
        n = int(input().strip())

    # Reading the image to be filtered
    image = imageio.imread(f'{input_image}')

    # Applying the filter according to the method selected
    if method == 1:
        result = limiarization(image, t0)

    elif method == 2:
        result = filter_1d(image, n, weights)

    elif method == 3:
        result = filter_2d(image, n, weights)
        result = limiarization(result, t0)

    elif method == 4:
        result = median_filter(image, n)

    # Calculating the error of the applied filter
    result_error = calculate_root_mean_sq_err(image, result)
    print(result_error)

