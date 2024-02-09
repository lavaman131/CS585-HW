import cv2
import numpy as np


def segment_image(binary_image: np.ndarray) -> np.ndarray:
    """
    Segments an image using the color model binary image conversion method.

    :param binary_image: A binary image to segment.
    """
    cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts[0], key=cv2.contourArea)
    return c
