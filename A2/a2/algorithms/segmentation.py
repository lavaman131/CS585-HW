import cv2
import numpy as np
from a2.data.preprocessing import adjust_gamma, color_model_binary_image_conversion


def find_max_countour(binary_image: np.ndarray) -> np.ndarray:
    """
    Helper function to segment a binary image by finding the max contour.

    :param binary_image: A binary image.
    :return: The max contour of the binary image.
    """
    cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts[0], key=cv2.contourArea)

    return c
