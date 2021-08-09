from typing import List
import numpy as np
import cv2.cv2 as cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

def myID() -> np.int:
    return 318554821


"""
Given two images, returns the Translation from im1 to im2
:param im1: Image 1
:param im2: Image 2
:param step_size: The image sample size:
:param win_size: The optical flow window size (odd number)
:return: Original points [[x,y]...], [[dU,dV]...] for each points
"""
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    derive_x = cv2.Sobel(im1, -1, 1, 0)
    derive_y = cv2.Sobel(im1, -1, 0, 1)
    derive_t = im1 - im2
    size = win_size // 2
    i_j = []
    u_v = []
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            x = derive_x[i - size:i + 1 + size, j - size: j + 1 + size]
            y = derive_y[i - size:i + 1 + size, j - size: j + 1 + size]
            t = derive_t[i - size:i + 1 + size, j - size: j + 1 + size]
            if x.size < win_size * win_size:
                break
            A = np.concatenate((x.reshape((win_size * win_size, 1)), y.reshape((win_size * win_size, 1))), axis=1)
            b = (t.reshape((win_size * win_size, 1)))
            g = np.linalg.eig(np.dot(A.T, A))[0]
            g = np.sort(g)
            if g[1] >= g[0] > 1 and (g[1] / g[0]) < 100:
                i_j.append(np.array([j, i]))
                u_v.append(np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b))
    return np.array(i_j), np.array(u_v)
"""
calculate gaussian
:return: Laplacian Pyramid (list of images) 
"""
def calcgaussian():
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    gaussian = cv2.getGaussianKernel(5, sigma)
    gaussian = gaussian * gaussian.transpose() * 4
    return gaussian

"""    
Creates a Laplacian pyramid
:param img: Original image
:param levels: Pyramid depth
:return: Laplacian Pyramid (list of images)
"""
def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    pyr = []
    gaussian = calcgaussian()
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]
    images = gaussianPyr(img, levels)
    new_image = img.copy()
    # go over all the images list
    for i in range(1, levels):
        expand = gaussExpand(images[i], gaussian)
        lap = new_image - expand
        pyr.append(lap)
        new_image = images[i]
    pyr.append(images[levels - 1])
    return pyr


"""    
Resotrs the original image from a laplacian pyramid
:param lap_pyr: Laplacian Pyramid
:return: Original image
"""
def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    lap_pyr.reverse()  # reverse the list - need to start from the end
    last = lap_pyr.pop(0)
    img = last
    gaussian = calcgaussian()
    for lap_img in lap_pyr:
        expand = gaussExpand(img, gaussian)
        img = expand + lap_img
    lap_pyr.insert(0, last)
    lap_pyr.reverse()  # return back
    return img

"""
Creates a Gaussian Pyramid
:param img: Original image
:param levels: Pyramid depth
:return: Gaussian pyramid (list of images)
"""
def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    h = pow(2, levels) * int((img.shape[0] // pow(2, levels)))
    w = pow(2, levels) * int((img.shape[1] // pow(2, levels)))
    img = img[:h, :w]
    temp = img.copy()
    pyr = [temp]  # level 0
    # all other levels - smaller images
    for i in range(levels - 1):
        temp = blurImage2(temp, 5)
        temp = temp[::2, ::2]
        pyr.append(temp)
    return pyr

def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    temp_gaussian = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian = temp_gaussian * temp_gaussian.transpose()
    return cv2.filter2D(in_image, -1, gaussian, borderType=cv2.BORDER_REPLICATE)


"""
Expands a Gaussian pyramid level one step up
:param img: Pyramid image at a certain level
:param gs_k: The kernel to use in expanding
:return: The expanded level
"""
def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        expand = np.zeros((2 * img.shape[0], 2 * img.shape[1], img.shape[2]), dtype=img.dtype)
    else:
        expand = np.zeros((2 * img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    expand[::2, ::2] = img
    return cv2.filter2D(expand, -1, gs_k, borderType=cv2.BORDER_REPLICATE)


"""
Blends two images using PyramidBlend method
:param img_1: Image 1
:param img_2: Image 2
:param mask: Blend mask
:param levels: Pyramid depth
:return: Blended Image
"""
def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    h = pow(2, levels) * (img_1.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_1.shape[1] // pow(2, levels))
    img_1 = img_1[:h, :w]

    h = pow(2, levels) * (img_2.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_2.shape[1] // pow(2, levels))
    img_2 = img_2[:h, :w]

    h = pow(2, levels) * (mask.shape[0] // pow(2, levels))
    w = pow(2, levels) * (mask.shape[1] // pow(2, levels))
    mask = mask[:h, :w]

    images_1 = laplaceianReduce(img_1, levels)
    images_2 = laplaceianReduce(img_2, levels)
    masks = gaussianPyr(mask, levels)
    blend = []
    for i in range(levels):
        blend.append(masks[i] * images_1[i] + (1 - masks[i]) * images_2[i])
    blend = laplaceianExpand(blend)
    # naive blend
    naive = img_1 * mask + (1 - mask) * img_2
    return naive, blend
