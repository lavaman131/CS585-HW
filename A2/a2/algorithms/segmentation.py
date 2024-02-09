import cv2
import numpy as np
from a2.data.preprocessing import adjust_gamma, color_model_binary_image_conversion


def segment_image(rgb_image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Segments an rgb image using the color model binary image conversion method.

    :param rgb_image: An rgb image to segment.
    :param gamma: The gamma value to use when adjusting the image.
    :return: The contours of the segmented image.
    """
    # sharpen kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    adjusted_image = cv2.filter2D(rgb_image, -1, kernel)

    adjusted_image = adjust_gamma(adjusted_image, gamma)
    binary_image = color_model_binary_image_conversion(adjusted_image)

    cnts = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts[0], key=cv2.contourArea)
    return c
