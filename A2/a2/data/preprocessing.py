import cv2
import numpy as np


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)


def gamma_binary_image_conversion(
    image: np.ndarray, thresh: int, gamma: float = 1.0
) -> np.ndarray:
    # adjust gamma
    adjusted_image = adjust_gamma(image, gamma)
    # extract red channel
    red_channel = adjusted_image[..., 2]
    # convert to binary image
    binary_image = cv2.threshold(red_channel, thresh, 255, cv2.THRESH_BINARY)[1]
    return binary_image


def color_model_binary_image_conversion(image: np.ndarray) -> np.ndarray:
    # formulas from: https://arxiv.org/pdf/1708.02694.pdf
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]

    rule_1 = (
        (H >= 0)
        & (H <= 50)
        & (S >= 0.23)
        & (S <= 0.68)
        & (R > 95)
        & (G > 40)
        & (B > 20)
        & (R > G)
        & (R > B)
        & (np.abs(R - G) > 15)
    )

    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    Y = ycrcb_image[:, :, 0]
    Cr = ycrcb_image[:, :, 1]
    Cb = ycrcb_image[:, :, 2]

    rule_2 = (
        (R > 95)
        & (G > 40)
        & (B > 20)
        & (R > G)
        & (R > B)
        & (np.abs(R - G) > 15)
        & (Cr > 135)
        & (Cb > 85)
        & (Y > 80)
        & (Cr <= 1.5862 * Cb + 20)
        & (Cr >= 0.3448 * Cb + 76.2069)
        & (Cr >= -4.5652 * Cb + 234.5652)
        & (Cr <= -1.15 * Cb + 301.75)
        & (Cr <= -2.2857 * Cb + 432.85)
    )

    mask = rule_1 | rule_2

    return mask.astype(np.uint8) * 255


# TODO: implement the following functions
# use template matching to recognize hand shapes
# use background subtraction to detect motion
# detect bounding boxes using horizontal and vertical projections
