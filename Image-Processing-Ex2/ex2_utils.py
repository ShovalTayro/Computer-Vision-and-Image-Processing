import cv2.cv2 as cv
import numpy as np

"""
Convolve a 1-D array with a given kernel
:param inSignal: 1-D array
:param kernel1: 1-D array as a kernel
:return: The convolved array
"""


def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    kernel = np.flip(kernel1)  # flip the vector
    a = np.pad(inSignal, (len(kernel) - 1, len(kernel) - 1), 'constant')
    conv = np.zeros(len(a) - len(kernel) + 1)
    for i in range(0, len(a) - len(kernel) + 1):
        conv[i] = np.multiply(a[i:i + len(kernel)], kernel).sum()  # multiply the values

    return conv


"""    
Convolve a 2-D array with a given kernel
:param inImage: 2D image
:param kernel2: A kernel
:return: The convolved image
"""


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    kernel = np.flip(kernel2)  # flip the vector
    im_pad = np.pad(inImage, (kernel.shape[0] // 2, kernel.shape[1] // 2), 'edge')  # padding the image
    conv = np.zeros((inImage.shape[0], inImage.shape[1]))
    for y in range(inImage.shape[0]):
        for x in range(inImage.shape[1]):
            conv[y, x] = (im_pad[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel).sum()  # convolution
    return conv


"""
Calculate gradient of an image    
:param inImage: Grayscale image
:return: (directions, magnitude,x_der,y_der)
"""


def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    kernel = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [0, -1, 0]])

    x_der = conv2D(inImage, kernel.transpose())
    y_der = conv2D(inImage, kernel)
    magnitude = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    directions = np.arctan2(y_der, x_der)
    return directions, magnitude, x_der, y_der


"""
Blur an image using a Gaussian kernel
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    assert (kernel_size % 2 == 1)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    mid = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - mid, j - mid
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    return conv2D(in_image, kernel)


"""
Blur an image using a Gaussian kernel using OpenCV built-in functions
:param inImage: Input image
:param kernelSize: Kernel size
:return: The Blurred image
"""


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    temp_gaussian = cv.getGaussianKernel(kernel_size, sigma)
    gaussian = temp_gaussian * temp_gaussian.transpose()
    return cv.filter2D(in_image, -1, gaussian, borderType=cv.BORDER_REPLICATE)


"""
Detects edges using the Sobel method
:param img: Input image
:param thresh: The minimum threshold for the edge response
:return: opencv solution, my implementation
"""


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    assert (1 >= thresh >= 0)
    x_der = conv2D(img, g_x)
    y_der = conv2D(img, g_y)

    implementation = np.sqrt(np.square(x_der) + np.square(y_der))
    implementation[implementation < thresh * 255] = 0
    implementation[implementation >= thresh * 255] = 1

    gradi_x = cv.Sobel(img, cv.CV_64F, 1, 0)
    gradi_y = cv.Sobel(img, cv.CV_64F, 0, 1)

    cv_solution = cv.magnitude(gradi_x, gradi_y)
    cv_solution[cv_solution < thresh * 255] = 0
    cv_solution[cv_solution >= thresh * 255] = 1
    return cv_solution, implementation


"""
Detecting edges using the "ZeroCrossing" method
:param img: Input image
:return: Edge matrix
"""


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    img = conv2D(img, laplacian)
    edge_matrix = np.zeros(img.shape)
    for i in range(img.shape[0] - (laplacian.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian.shape[1] - 1)):
            if img[i][j] == 0:
                if (img[i][j - 1] < 0 and img[i][j + 1] < 0) or (
                        img[i][j - 1] < 0 and img[i][j + 1] > 0) or (
                        img[i - 1][j] < 0 and img[i + 1][j] > 0) or (
                        img[i - 1][j] > 0 and img[i + 1][j] < 0):
                    edge_matrix[i][j] = 1
            if img[i][j] < 0:
                if (img[i][j - 1] > 0) or (
                        img[i][j + 1] > 0) or (
                        img[i - 1][j] > 0) or (
                        img[i + 1][j] > 0):
                    edge_matrix[i][j] = 1
    return edge_matrix


"""
Detecting edges using the "ZeroCrossingLOG" method
:param img: Input image
:return: :return: Edge matrix
"""


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    blur = cv.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(blur)


"""
Detecting edges usint "Canny Edge" method
:param img: Input image
:param thrs_1: T1
:param thrs_2: T2
:return: opencv solution, my implementation
"""
def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    # Sobel edge detection
    magnitude = np.sqrt(np.power(cv.Sobel(img, -1, 0, 1), 2) + np.power(cv.Sobel(img, -1, 1, 0), 2))
    directions = np.arctan2(cv.Sobel(img, -1, 0, 1), cv.Sobel(img, -1, 1, 0))
    # NMS
    NMS = non_max_suppression(magnitude, directions)
    for i in range(NMS.shape[0]):
        for j in range(NMS.shape[1]):
            if NMS[i][j] <= thrs_2:
                NMS[i][j] = 0
            elif thrs_2 < NMS[i][j] < thrs_1:
                n = NMS[i - 1:i + 2, j - 1: j + 2]  # mark as edge?
                if n.max() < thrs_1:
                    NMS[i][j] = 150
                else:
                    NMS[i][j] = 255
            else:
                NMS[i][j] = 255

    # check potential edges
    for i in range(NMS.shape[0]):
        for j in range(NMS.shape[1]):
            if NMS[i][j] == 150:
                n = NMS[i - 1:i + 2, j - 1: j + 2] # mark as edge?
                if n.max() < thrs_1:
                    NMS[i][j] = 0
                else:
                    NMS[i][j] = 255

    cv_solution = cv.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    return cv_solution, NMS

def non_max_suppression(img: np.ndarray, D: np.ndarray) -> (np.ndarray):
    mat = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    angles = np.rad2deg(D)
    angles[angles < 0] += 180
    # go over the matrix
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            x = 255
            y = 255
            # angle 0
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                x = img[i, j + 1]
                y = img[i, j - 1]
            # angle 45
            elif 22.5 <= angles[i, j] < 67.5:
                x = img[i + 1, j - 1]
                y = img[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angles[i, j] < 112.5:
                x = img[i + 1, j]
                y = img[i - 1, j]
            # angle 135
            elif 112.5 <= angles[i, j] < 157.5:
                x = img[i - 1, j - 1]
                y = img[i + 1, j + 1]

            if (img[i, j] >= x) and (img[i, j] >= y):
                mat[i, j] = img[i, j]
            else:
                mat[i, j] = 0

    return mat


"""
Find Circles in an image using a Hough Transform algorithm extension
:param I: Input image
:param minRadius: Minimum circle radius
:param maxRadius: Maximum circle radius
:return: A list containing the detected circles,
[(x,y,radius),(x,y,radius),...]
"""

def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.Canny(img.astype(np.uint8), 50, 100)
    edges = np.argwhere(img > 0)
    # accumulator array
    acc = np.zeros((max_radius, img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius))
    sigma = np.arange(0, 360) * np.pi / 180

    for r in range(round(min_radius), round(max_radius)):
        temp_circle = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (i, j) = (r + 1, r + 1)  # the center
        for angle in sigma:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            temp_circle[i + x, j + y] = 1
        constant = np.argwhere(temp_circle).shape[0]
        for x, y in edges:
            acc[r, x - i + max_radius:x + i + max_radius, y - j + max_radius:y + j + max_radius] += temp_circle
        acc[r][acc[r] < 7 * constant / r] = 0  # threshold

    # accumulator array for the info of the circle
    info = np.zeros((max_radius, img.shape[0] + 2 * max_radius, img.shape[1] + 2 * max_radius))

    for r, x, y in np.argwhere(acc):
        c_range = acc[r - 15 : r + 15, x - 15 : x + 15, y - 15 : y + 15]
        a, b, c = np.unravel_index(np.argmax(c_range), c_range.shape)
        info[r + (a - 15), x + (a - 15), y + (c - 15)] = 1

    circles = np.argwhere(info[:, max_radius:-max_radius, max_radius:-max_radius])

    return circles