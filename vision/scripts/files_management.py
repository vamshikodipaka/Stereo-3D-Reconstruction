import csv
import numpy as np
import cv2


def get_projection_matrices():
    """Frame Calibration Holder
    3x4    p_left, p_right      Camera P matrix. Contains extrinsic and intrinsic parameters.
    """
    p_left = np.array([[ 1.,   0.,   3.1969315809942782e+02, 0.],
                       [ 0.,   1.,   1.7933051140233874e+02, 0.],
                       [ 0.,   0.,   1.                    , 0. ]])
    p_right = np.array([[1.,   0.,   3.1969315809942782e+02, 0.],
                        [0.,   1.,   1.7933051140233874e+02, 4.0590762653907342e-01],
                        [0.,   0.,   1.,                     0. ]])
    return p_left, p_right 


def read_left_image():
    return cv2.imread("stereo_set/imL.png")[...,::-1]


def read_right_image():
    return cv2.imread("stereo_set/imR.png")[...,::-1]


def get_obstacle_image():
    img_left_colour = read_left_image()
    return img_left_colour[479:509, 547:593, :]
