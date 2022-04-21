# Henrique Bonini de Britto Menezes, 8956690, scc5830, 01/2022, Assignment 1 - Image Generation
import numpy as np
import imageio
import random

import time


def pixel_f1(x, y, q):
    '''Function 1 of image pixels generation

    Arguments:
        x: row coordinate for a given pixel
        y: column coordinate for a given pixel
        q: mathematical function parameter Q
    
    Returns:
        A pixel generated according to the selected function
    '''
    return (x*y + 2*y)

def pixel_f2(x, y, q):
    '''Function 2 of image pixels generation

    Arguments:
        x: row coordinate for a given pixel
        y: column coordinate for a given pixel
        q: mathematical function parameter Q
    
    Returns:
        A pixel generated according to the selected function
    '''
    return np.abs(np.cos(x/q) + 2 * np.sin(y/q))

def pixel_f3(x, y, q):
    '''Function 3 of image pixels generation

    Arguments:
        x: row coordinate for a given pixel
        y: column coordinate for a given pixel
        q: mathematical function parameter Q
    
    Returns:
        A pixel generated according to the selected function
    '''
    return np.abs(3 * (x/q) -  np.cbrt(y/q))

def pixel_f4(x, y, q):
    '''Function 4 of image pixels generation

    Arguments:
        x: row coordinate for a given pixel
        y: column coordinate for a given pixel
        q: mathematical function parameter Q
    
    Returns:
        A pixel generated according to the selected function
    '''
    return random.random()

# Funtions dictionary for a cleaner code
funct = {
    1: pixel_f1,
    2: pixel_f2,
    3: pixel_f3,
    4: pixel_f4,
}

def scene_f5(scene, size_c):
    '''Funcion 5 for scene generation, it needs to access the entire matrix at any given time

    Arguments:
        scene: Pre-initialised ndarray matrix with zeroes
        size_c: size of both of the matrix' dimentions
    
    Returns:
        The generated image in-place
    '''
    start_time = time.time()
    x = 0
    y = 0
    scene[0, 0] = 1
    for _ in range(np.power(size_c, 2) + 1):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        x = np.mod(x + dx, size_c)
        y = np.mod(y + dy, size_c)
        scene[x, y] = 1
    print(f'Total generation time: {time.time() - start_time}')
    return scene


def generate_scene_image(f, size_c, q, seed):
    scene = np.zeros((size_c, size_c))
    random.seed(seed)
    if f == 5:
        scene = scene_f5(scene, size_c)
    else:
        for x in range(size_c):
            for y in range(size_c):
                scene[x, y] = funct[f](x, y, q)
    
    imax = np.max(scene)
    imin = np.min(scene)

    scene_norm = (scene - imin)/(imax - imin)
    scene_norm = (scene_norm * 65535).astype(np.uint16)
    return scene_norm

def generate_digital_image(scene, size_c, size_n, bytes):
    dig_image = np.zeros((size_n, size_n))
    downsample_step = size_c // size_n
    i = 0
    j = 0
    for x in range(0, size_c, downsample_step):
        for y in range(0, size_c, downsample_step):
            try:
                dig_image[i, j] = scene[x, y]
            except IndexError:
                break
            j += 1
        i += 1
        j = 0
    
    imax = np.max(dig_image)
    imin = np.min(dig_image)
    dig_image_norm = (dig_image - imin)/(imax - imin)
    dig_image_norm = (dig_image_norm * 255).astype(np.uint8)

    dig_image_shifted = np.right_shift(dig_image_norm, (8 - bytes))

    return dig_image_shifted

def calculate_root_sq_err(ref_image, dig_image):
    subtracted_image = dig_image.astype(float) - ref_image.astype(float)
    squared_image = np.square(subtracted_image)
    err = np.sum(squared_image)
    
    return np.sqrt(err)

def main():
    ref_image_path = str(input()).rstrip()
    size_c = int(input())
    f = int(input())
    q = float(input())
    size_n = int(input())
    b = int(input())
    seed = int(input())

    scene_image = generate_scene_image(f, size_c, q, seed)
    dig_image = generate_digital_image(scene_image, size_c, size_n, b)

    # imageio.imwrite('scene.png', scene_image)
    # imageio.imwrite('dig.png', dig_image)

    ref_image = np.load(ref_image_path).astype(np.uint8)

    mse = calculate_root_sq_err(ref_image, dig_image)

    print(f'{mse:.4f}')

if __name__ == "__main__":
    main()
