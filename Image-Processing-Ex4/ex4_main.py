# ps2
import os
import numpy as np
from ex4_utils import *
import cv2


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():

    print(myID())

    ## 1-a
    # Read images
    i = 1
    L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), 0) / 255.0
    R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), 0) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (0, 5), method=disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (10, 150), method=disparityNC)

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)
    print(h)
    print("error =", error)

    dst = cv2.imread(os.path.join('input', 'billBoard.jpg'), 0) / 255.0
    src = cv2.imread(os.path.join('input', 'car.jpg'), 0) / 255.0

    warpImag(src, dst)


if __name__ == '__main__':
    main()
