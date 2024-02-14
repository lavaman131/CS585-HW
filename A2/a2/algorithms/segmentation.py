import cv2
import numpy as np
from typing import Sequence, Optional


def find_max_countour(cnts: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """
    Helper function to segment a binary image by finding the max contour.

    :param cnts: A sequence of contours.
    :return: The max contour of the binary image.
    """
    return max(cnts[0], key=cv2.contourArea) if cnts else None
