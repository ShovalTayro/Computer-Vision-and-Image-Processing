"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

"""
Return my ID (not the friend's ID I copied from)
:return: int
"""
def myID() -> np.int:
    return 318554821


"""
Reads an image, and returns the image converted as requested
:param filename: The path to the image
:param representation: GRAY_SCALE or RGB
:return: The image object
"""
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    src = cv2.imread(filename)
    if representation == LOAD_RGB:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    else:  # GRAY_SCALE
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Normalization
    img = (img - img.min()) / (img.max() - img.min())
    return img


"""
Reads an image as RGB or GRAY_SCALE and displays it    
:param filename: The path to the image
:param representation: GRAY_SCALE or RGB
:return: None
"""
def imDisplay(filename: str, representation: int):
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:  # GRAY_SCALE
        plt.gray()
    plt.imshow(img)
    plt.show()


"""
Converts an RGB image to YIQ color space
:param imgRGB: An Image in RGB
:return: A YIQ in image color space
"""
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    return np.dot(imgRGB, mat.T.copy())


"""
Converts an YIQ image to RGB color space
:param imgYIQ: An Image in YIQ
:return: A RGB in image color space
"""
def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    # Invertible matrix
    invertible_mat = np.linalg.inv(mat)
    return np.dot(imgYIQ, invertible_mat.T.copy())


# Image is a RGB or GRAY
def isRGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    is_color = False
    # color image is a 3D matrix (MxNx3)
    if len(imgOrig.shape) == 3:
        is_color = True
        image_YIQ = transformRGB2YIQ(imgOrig)
        imgOrig = np.copy(image_YIQ[:, :, 0])  # Y channel
        return is_color, imgOrig, image_YIQ

    return is_color, imgOrig, None


"""
Equalizes the histogram of an image        
:param imgOrig: Original Histogram
:ret
"""
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    # Image is a RGB or GRAY
    is_RGB, img, image_YIQ = isRGB(imgOrig)

    # Calculate the image histogram (range = [0, 255])
    img = (np.around(img * 255)).astype('uint8')
    hist_original = np.histogram(img.flatten(), bins=256, range=[0, 255])[0]

    # Calculate the normalized Cumulative Sum
    cum_sum = np.cumsum(hist_original)

    # Create a LookUpTable(LUT)
    image_Eq = cum_sum[img]
    image_Eq = (image_Eq * 255.0 / image_Eq.max()).astype('uint8')
    hist_Eq = np.histogram(image_Eq.flatten(), bins=256, range=[0, 255])[0]

    if is_RGB:  # Return image to RGB
        image_YIQ[:, :, 0] = image_Eq / (image_Eq.max() - image_Eq.min())
        image_Eq = transformYIQ2RGB(image_YIQ)

    return image_Eq, hist_original, hist_Eq

"""
Quantized an image in to **nQuant** colors        
:param imOrig: The original image (RGB or Gray scale)        
:param nQuant: Number of colors to quantize the image to
:param nIter: Number of optimization loops
:return: (List[qImage_i],List[error_i])
"""

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    # Image is a RGB or GRAY
    is_RGB, image, image_YIQ = isRGB(imOrig)
    img = image * 255
    original_hist = np.histogram(img.flatten(), 256)[0]

    # Finding Z & Q
    z = np.arange(0, 256, int(255 / nQuant))  # From 0 to 255 with steps of 255/nQuant
    z[nQuant] = 255  # Last element is 255

    image_list = []
    error_list = []

    for k in range(0, nIter):
        # Init Q
        q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=original_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
        q = np.round(q).astype(int)

        # Move boundaries to be in the middle of two means
        for j in range(1, nQuant):
            z[j] = np.round((q[j - 1] + q[j]) / 2)

        new_img = img.copy()
        # Update value
        for i in range(1, nQuant + 1):
            if is_RGB:
                new_img[(new_img > z[i - 1]) & (new_img < z[i])] = q[i - 1]
            else:
                new_img[(new_img >= z[i - 1]) & (new_img < z[i])] = q[i - 1]

        # Calculate the error
        MSE = pow(np.power(img - new_img, 2).sum(), 0.5)/img.size

        if is_RGB:
            image_YIQ[:, :, 0] = new_img / (new_img.max() - new_img.min())
            new_img = transformYIQ2RGB(image_YIQ)

        image_list.append(new_img)
        error_list.append(MSE)
    return image_list, error_list
