import cv2
import math
import numpy as np
from enum import Enum, unique

def get_blob_barycenter(contour):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy

def relative_pos_from_pixels(img, x_pixels, y_pixels):
    """
    Transform a pixels position to a relative position

    :param img: Image where the object is detected
    :type img: numpy.array
    :param x_pixels: coordinate X
    :type x_pixels: int
    :param y_pixels: coordinate Y
    :type y_pixels: int
    :return: X relative, Y relative
    :rtype: float, float
    """
    return float(x_pixels) / img.shape[1], float(y_pixels) / img.shape[0]

def get_contour_angle(contour):
    """
    Return orientation of a contour according to the smallest side
    in order to be well oriented for gripper

    :param contour: contour
    :type contour: OpenCV Contour
    :return: Angle in radians
    :rtype: float
    """
    rotrect = cv2.minAreaRect(contour)
    angle = rotrect[-1]
    size1, size2 = rotrect[1][0], rotrect[1][1]
    ratio_size = float(size1) / float(size2)
    if 1.25 > ratio_size > 0.75:
        if angle < -45:
            angle = 90 + angle
    else:
        if size1 < size2:
            angle = angle + 180
        else:
            angle = angle + 90

        if angle > 90:
            angle = angle - 180

    return math.radians(angle)


def biggest_contour_finder(img):
    res = biggest_contours_finder(img, nb_contours_max=1)
    if not res:
        return res
    else:
        return res[0]

def biggest_contours_finder(img, nb_contours_max=3):
    """
    Function to find the biggest contour in an binary image

    :param img: Binary Image
    :type img: numpy.array
    :param nb_contours_max: maximal number of contours which will be returned
    :type nb_contours_max: int
    :return: biggest contours found
    :rtype: list[OpenCV Contour]
    """
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    if not contours:
        return []
    contours_area = list()
    for cnt in contours:
        contours_area.append(cv2.contourArea(cnt))
    biggest_contours = []
    le = len(contours_area)
    if nb_contours_max > le:
        nb_contours = le
        id_contours_sorted_init = list(range(nb_contours))
    else:
        nb_contours = nb_contours_max
        id_contours_sorted_init = np.argpartition(contours_area, -nb_contours)[-nb_contours:]
    id_contours_sorted = [x for x in sorted(id_contours_sorted_init, key=lambda idi: -contours_area[idi])]

    for i in range(nb_contours):
        id_used = id_contours_sorted[i]

        if contours_area[id_used] < 400:
            break

        biggest_contours.append(contours[id_used])
    return biggest_contours

def uncompress_image(compressed_image):
    """
    Take a compressed img and return an OpenCV image

    :param compressed_image: compressed image
    :type compressed_image: str
    :return: OpenCV image
    :rtype: numpy.array
    """
    np_arr = np.fromstring(compressed_image, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def undistort_image(img, mtx, dist):
    """
    Use camera intrinsics to undistort raw image

    :param img: Raw Image
    :type img: numpy.array
    :param mtx: Camera Intrinsics matrix
    :type mtx: list[list[float]]
    :param dist: Distortion Coefficient
    :type dist: list[list[float]]
    :return: Undistorted image
    :rtype: numpy.array
    """
    return cv2.undistort(src=img, cameraMatrix=mtx, distCoeffs=dist)

# Image Processing
@unique
class ColorHSV(Enum):
    """
    MIN HSV, MAX HSV, Invert Hue (bool)
    """
    BLUE = [90, 50, 85], [125, 255, 255], False
    RED = [15, 80, 75], [170, 255, 255], True
    GREEN = [40, 60, 75], [85, 255, 255], False
    ANY = [0, 50, 100], [179, 255, 255], False

def threshold_hsv(img, list_min_hsv, list_max_hsv, reverse_hue=False, use_s_prime=False):
    """
    Take BGR image (OpenCV imread result) and return thresholded image
    according to values on HSV (Hue, Saturation, Value)
    Pixel will worth 1 if a pixel has a value between min_v and max_v for all channels

    :param img: image BGR if rgb_space = False
    :type img: numpy.array
    :param list_min_hsv: list corresponding to [min_value_H,min_value_S,min_value_V]
    :type list_min_hsv: list[int]
    :param list_max_hsv: list corresponding to [max_value_H,max_value_S,max_value_V]
    :type list_max_hsv: list[int]
    :param use_s_prime: True if you want to use S channel as S' = S x V else classic
    :type use_s_prime: bool
    :param reverse_hue: Useful for Red color cause it is at both extremum
    :type reverse_hue: bool
    :return: threshold image
    :rtype: numpy.array
    """
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if use_s_prime:
        frame_hsv[:, :, 1] = (1. / 255) * frame_hsv[:, :, 1] * frame_hsv[:, :, 2].astype(np.uint8)

    if not reverse_hue:
        return cv2.inRange(frame_hsv, tuple(list_min_hsv), tuple(list_max_hsv))
    else:
        list_min_v_c = list(list_min_hsv)
        list_max_v_c = list(list_max_hsv)
        lower_bound_red, higher_bound_red = sorted([list_min_v_c[0], list_max_v_c[0]])
        list_min_v_c[0], list_max_v_c[0] = 0, lower_bound_red
        low_red_im = cv2.inRange(frame_hsv, tuple(list_min_v_c), tuple(list_max_v_c))
        list_min_v_c[0], list_max_v_c[0] = higher_bound_red, 179
        high_red_im = cv2.inRange(frame_hsv, tuple(list_min_v_c), tuple(list_max_v_c))
        return cv2.addWeighted(low_red_im, 1.0, high_red_im, 1.0, 0)