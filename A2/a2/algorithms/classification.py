import cv2
import numpy as np


def template_match(image: np.ndarray, binary_templates: np.ndarray) -> np.ndarray:
    """
    Uses template matching to recognize hand shapes.

    :param image: The image to recognize hand shapes in.
    :param template: The template to match.
    :return: The match.
    """
    # score all templates in the image
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
