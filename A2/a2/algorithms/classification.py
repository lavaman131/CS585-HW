import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def template_match(
    row: pd.Series,
    image: np.ndarray,
    binary_template: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> Tuple[float, int]:
    pred = 1
    max_score = -1
    for scale in scales:
        new_width, new_height = (
            int(binary_template.shape[1] * scale),
            int(binary_template.shape[0] * scale),
        )
        binary_template_resized = cv2.resize(
            binary_template.copy(),
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        for angle in rotations:
            M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
            binary_template_rotated = cv2.warpAffine(
                binary_template_resized.copy(),
                M,
                (new_width, new_height),
            )
            matches = cv2.matchTemplate(
                image, binary_template_rotated, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, _ = cv2.minMaxLoc(matches)
            if score > max_score:
                max_score = score
                pred = row.label

    return max_score, pred


def template_match_classify(
    image: np.ndarray,
    template_images_dir: str,
    image_metadata: pd.DataFrame,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> Dict[str, float]:
    """
    Uses template matching to recognize hand shapes.

    :param image: The image to recognize hand shapes in.
    :param template_images_dir: The directory containing the binary template images.
    :param image_metadata: The metadata of the binary template images.
    :param scales: The scales to resize the binary template images to.
    :param rotations: The rotations to rotate the binary template images by.
    :return: The predicted label.
    """
    max_pred = 1
    max_score = -1
    for i, row in image_metadata.iterrows():
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
        score, pred = template_match(
            row, image, binary_template_image, scales, rotations
        )
        score_flipped, pred_flipped = template_match(
            row, image, cv2.flip(binary_template_image, 1), scales, rotations
        )

        if score_flipped > score:
            score = score_flipped
            pred = pred_flipped

        if score > max_score:
            max_score = score
            max_pred = pred

    return {"pred": max_pred, "score": max_score}
