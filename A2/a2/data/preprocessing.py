import cv2
import numpy as np


def post_process_binary_image(
    binary_image: np.ndarray, max_countour: np.ndarray
) -> np.ndarray:
    """
    Post processes a binary image by finding the max contour and filling everything outside of the contour with white.

    :param binary_image: A binary image.
    :param max_countour: The max contour of the binary image.
    :return: The post processed image.
    """

    # fill everything outside of the contour with white
    post_processed_image = np.full_like(binary_image, 255, dtype=np.uint8)
    post_processed_image = cv2.fillPoly(post_processed_image, [max_countour], 0)  # type: ignore

    return post_processed_image


def color_model_binary_image_conversion(
    rgb_image: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Converts an RGB image to a binary image using the color model method from: https://arxiv.org/pdf/1708.02694.pdf
    :param rgb_image: An RGB image to convert to a binary image.
    :param gamma: Gamma value for adjusting the brightness of the image.
    :return: A binary image.
    """
    # sharpen kernel
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # adjusted_image = cv2.filter2D(rgb_image, -1, kernel)

    adjusted_image = adjust_gamma(rgb_image, gamma)

    B = adjusted_image[:, :, 0]
    G = adjusted_image[:, :, 1]
    R = adjusted_image[:, :, 2]

    hsv_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]

    rule_1 = (
        (H >= 0)  # type: ignore
        & (H <= 50)  # type: ignore
        & (S >= 0.23)  # type: ignore
        & (S <= 0.68)  # type: ignore
        & (R > 95)
        & (G > 40)
        & (B > 20)
        & (R > G)
        & (R > B)
        & (np.abs(R - G) > 15)
    )

    ycrcb_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2YCrCb)

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
        & (Cr > 135)  # type: ignore
        & (Cb > 85)  # type: ignore
        & (Y > 80)  # type: ignore
        & (Cr <= 1.5862 * Cb + 20)  # type: ignore
        & (Cr >= 0.3448 * Cb + 76.2069)  # type: ignore
        & (Cr >= -4.5652 * Cb + 234.5652)  # type: ignore
        & (Cr <= -1.15 * Cb + 301.75)  # type: ignore
        & (Cr <= -2.2857 * Cb + 432.85)  # type: ignore
    )

    binary_image = (rule_1 | rule_2).astype(np.uint8) * 255

    return binary_image


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)
