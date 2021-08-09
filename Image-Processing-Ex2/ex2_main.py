from ex2_utils import *
import matplotlib.pyplot as plt
import time

def conv1Demo():
    print("----- Conv1 -----")
    a = np.arange(10000.0)
    b = np.array([0, 1, 0.5])
    print("conv1Demo:", np.sum(np.convolve(a, b) - conv1D(a, b)))


def conv2Demo(img: np.ndarray):
    print("----- Conv2 -----")
    kernel = np.ones((5, 5))
    kernel /= kernel.sum()
    cv2_img = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REPLICATE)
    start_time = time.time()
    conv_img = conv2D(img, kernel)
    print("Time:%.2f" % (time.time() - start_time))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(cv2_img)
    ax[0].set_title("conv2D cv2")
    ax[1].imshow(conv_img)
    ax[1].set_title("mine")
    plt.show()


def derivDemo(img: np.ndarray):
    print("----- Derivative -----")
    start_time = time.time()
    direction, magnitude, x_der, y_der = convDerivative(img)
    print("Time:%.2f" % (time.time() - start_time))
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle('Derivative', fontsize=16)
    ax1.imshow(x_der)
    ax1.set_title('x_der')
    ax2.imshow(y_der)
    ax2.set_title('y_der')
    ax3.imshow(magnitude)
    ax3.set_title('magnitude')
    ax4.imshow(direction)
    ax4.set_title('direction')
    plt.show()


def blurDemo(img: np.ndarray):
    print("----- Blur_image -----")
    start_time = time.time()
    size = 23
    blur1 = blurImage1(img, size)
    blur2 = blurImage2(img, size)
    print("Time:%.2f" % (time.time() - start_time))
    f, ax = plt.subplots(1, 2)
    f.suptitle('blur', fontsize=16)
    plt.title("blur")
    ax[0].imshow(blur1, cmap="gray")
    ax[0].set_title('mine')
    ax[1].imshow(blur2, cmap="gray")
    ax[1].set_title('blurImage cv2')
    plt.show()


def edgeDemo (img1: np.ndarray, img2: np.ndarray):
    sobel(img1)
    zero_cross(img2)
    canny(img1)

def sobel(img: np.ndarray):
    print("----- Sovel -----")
    start_time = time.time()
    cv2_edge_img, edge_img = edgeDetectionSobel(img, 0.4)
    print("Time:%.2f" % (time.time() - start_time))
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2_edge_img)
    ax[0].set_title("Sobel cv2")
    ax[1].imshow(edge_img)
    ax[1].set_title("mine")
    plt.show()


def zero_cross(img: np.ndarray):
    print("----- Zero_cross -----")
    start_time = time.time()
    edge_imgSimple = edgeDetectionZeroCrossingSimple(img)
    print("Time:%.2f" % (time.time() - start_time))
    plt.imshow(edge_imgSimple)
    plt.title("zero crossing simple")
    plt.show()

    print("----- Zero_cross_LOG -----")
    start_time2 = time.time()
    edge_imgLog = edgeDetectionZeroCrossingLOG(img)
    print("Time:%.2f" % (time.time() - start_time2))
    plt.imshow(edge_imgLog)
    plt.title("zero crossing LOG")
    plt.show()


def canny(img: np.ndarray):
    print("----- Canny -----")
    start_time = time.time()
    cv2_canny, canny_image = edgeDetectionCanny(img, 100, 50)
    print("Time:%.2f" % (time.time() - start_time))
    f, ax = plt.subplots(1, 2)
    f.suptitle('canny', fontsize=16)
    ax[0].imshow(cv2_canny, cmap="gray")
    ax[0].set_title('canny cv2')
    ax[1].imshow(canny_image, cmap="gray")
    ax[1].set_title('mine')
    plt.show()

def houghDemo(img: np.ndarray):
    print("----- Hough_circle -----")
    start_time = time.time()
    hough_img = houghCircle(img, 90, 95)
    print("Time:%.2f" % (time.time() - start_time))

    fig = plt.figure()
    plt.imshow(img)
    circles = []
    for r, x, y in hough_img:
        circles.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot().add_artist(circles[-1])
    plt.title("hough circle")
    plt.show()

def main():
    boxman = cv.imread("boxman.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    circle = cv.imread("circles.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    monkey = cv.imread("codeMonkey.jpeg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    beach = cv.imread("beach.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)

    conv1Demo()
    conv2Demo(boxman)
    derivDemo(beach)
    blurDemo(beach)
    edgeDemo(boxman,monkey)
    houghDemo(circle)

if __name__ == '__main__':
    main()
