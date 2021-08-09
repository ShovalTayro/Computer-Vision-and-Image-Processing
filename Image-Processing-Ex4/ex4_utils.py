from pylab import *
from scipy.ndimage import *

"""
img_l: Left image
img_r: Right image
range: Minimum and Maximum disparity range. Ex. (10,80)
k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
return: Disparity map, disp_map.shape = Left.shape
"""


"""
Return my ID
:return: int
"""
def myID() -> np.int:
    return 318554821

def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    height, width = img_l.shape  # assume that both images are same size
    disp_map = np.zeros((height, width, disp_range[1]))

    # average per pixel
    avg_l = np.zeros((height, width))
    avg_r = np.zeros((height, width))
    filters.uniform_filter(img_l, k_size, avg_l)
    filters.uniform_filter(img_r, k_size, avg_r)

    # normalized image
    normalize_l = img_l - avg_l
    normalize_r = img_r - avg_r

    # shift right
    for s in range(disp_range[1]):
        img_s_r = np.roll(normalize_r, s)
        filters.uniform_filter(normalize_l * img_s_r, k_size, disp_map[:, :, s])
        # minimum SSD
        disp_map[:, :, s] = disp_map[:, :, s] ** 2

    depth = np.argmax(disp_map, axis=2)
    return depth


"""
img_l: Left image
img_r: Right image
range: Minimum and Maximum disparity range. Ex. (10,80)
k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
return: Disparity map, disp_map.shape = Left.shape
"""


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    height, width = img_r.shape
    disp_map = np.zeros((height, width, disp_range[1]))

    # average per pixel
    avg_l = np.zeros((height, width))
    avg_r = np.zeros((height, width))
    filters.uniform_filter(img_l, k_size, np.zeros((height, width)))
    filters.uniform_filter(img_r, k_size, np.zeros((height, width)))

    # normalized image
    normalize_l = img_l - avg_l
    normalize_r = img_r - avg_r

    shift = np.zeros((height, width))
    shift_l = np.zeros((height, width))
    shift_r = np.zeros((height, width))
    filters.uniform_filter(normalize_l * normalize_l, k_size, shift_l)

    # shift right
    for s in range(disp_range[1]):
        img_s_r = np.roll(normalize_r, s - disp_range[0])
        filters.uniform_filter(normalize_l * img_s_r, k_size, shift)
        filters.uniform_filter(img_s_r * img_s_r, k_size, shift_r)
        # NCC
        disp_map[:, :, s] = shift / np.sqrt(shift_l * shift_l)

    depth = np.argmax(disp_map, axis=2)
    return depth


"""
Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
returns the homography and the error between the transformed points to their
destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))
src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]
return: (Homography matrix shape:[3,3], Homography error)
"""


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    A = np.zeros((2 * len(src_pnt), 9))
    index = 0
    for i in range(0, len(src_pnt)):
        x, y = src_pnt[i][0], src_pnt[i][1]
        u, v = dst_pnt[i][0], dst_pnt[i][1]
        A[index] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[index + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]
        index = index + 2

    # SVD
    U, S, V_T = np.linalg.svd(A)
    h = V_T[-1] / V_T[-1, -1]
    h = h.reshape(3, 3)
    error = 0
    for i in range(len(src_pnt)):
        x = src_pnt[i, 0]
        y = src_pnt[i, 1]
        # convert the coordinates to homogeneous coordinates
        homogeneous_coordinates = np.array([x, y, 1])
        homogeneous_coordinates = h.dot(homogeneous_coordinates)
        homogeneous_coordinates /= homogeneous_coordinates[2]
        homogeneous_coordinates = homogeneous_coordinates[0:-1]
        # compute the error
        error += np.sqrt(sum(homogeneous_coordinates - dst_pnt[i]) ** 2)
    return h, error

"""
Displays both images, and lets the user mark 4 or more points on each image. 
Then calculates the homography and transforms the source image on to the destination image. 
Then transforms the source image onto the destination image and displays the result.
src_img: The image that will be 'pasted' onto the destination image.
dst_img: The image that the source image will be 'pasted' on.
output:
    None.
"""
def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    dst_p = []
    fig1 = plt.figure()
    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, 'r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display  first image
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)
    src_p = []
    fig2 = plt.figure()
    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))
        plt.plot(x, y, 'r')
        src_p.append([x, y])
        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display second image
    cid2 = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    # compute the homography
    h = computeHomography(src_p, dst_p)[0]
    # transforms the source image on to the destination image.
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            homogeneous_coordinates = np.array([j, i, 1])
            homogeneous_coordinates = h.dot(homogeneous_coordinates)
            homogeneous_coordinates /= homogeneous_coordinates[2]
            # transforms the source image onto the destination image and displays the result.
            dst_img[int(homogeneous_coordinates[1]), int(homogeneous_coordinates[0])] = src_img[i, j]
    plt.imshow(dst_img)
    plt.show()