import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def template_match_classify(
    image: np.ndarray, template_images_dir: str, image_metadata: pd.DataFrame
) -> int:
    """
    Uses template matching to recognize hand shapes.

    :param image: The image to recognize hand shapes in.
    :param template_images_dir: The directory containing the binary template images.
    :param image_metadata: The metadata of the binary template images.
    :return: The predicted label.
    """
    pred = 1
    max_score = -1
    for _, row in image_metadata.iterrows():
        binary_template_image = cv2.imread(
            str(Path(template_images_dir).joinpath(row.image_name)),
            flags=cv2.IMREAD_GRAYSCALE,
        )
        # make binary image dimensions match the input image
        binary_template_image = cv2.resize(
            binary_template_image,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        # check slight rotations of the template image [-20, 20] with step 5
        for angle in range(-20, 25, 5):
            M = cv2.getRotationMatrix2D(
                (image.shape[1] // 2, image.shape[0] // 2), angle, 1
            )
            binary_template_rotated = cv2.warpAffine(
                binary_template_image, M, (image.shape[1], image.shape[0])
            )
            matches = cv2.matchTemplate(
                image, binary_template_rotated, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, _ = cv2.minMaxLoc(matches)
            if score > max_score:
                max_score = score
                pred = row.label

    return pred
