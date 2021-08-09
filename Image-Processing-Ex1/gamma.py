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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

def on_trackbar(x: int):
    pass

def gammaCorrection(img: np.ndarray, gamma: float) -> np.ndarray:
    img = img / 255.0
    img_array = np.power(img, gamma)  # for every i in img -> i ** gamma
    return img_array

"""
GUI for gamma correction
:param img_path: Path to the image
:param rep: grayscale(1) or RGB(2)
:return: None
"""
def gammaDisplay(img_path: str, rep: int):
    src = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    name = 'Gamma Correction'
    cv2.namedWindow(name)
    cv2.createTrackbar('Gamma', name, 1, 100, on_trackbar)
    while True:
        gamma = cv2.getTrackbarPos('Gamma', name)
        gamma = gamma / 50
        # make gamma correction
        img = gammaCorrection(src, gamma)
        cv2.imshow(name, img)
        key = cv2.waitKey(1000)
        if key == 27:  # ESC
            break
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(name)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
