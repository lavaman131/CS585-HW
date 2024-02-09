import cv2
import numpy as np


def template_match(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Uses template matching to recognize hand shapes.

    :param image: The image to recognize hand shapes in.
    :param template: The template to match.
    :return: The match.
    """
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
