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
    max_pred = 1
    max_score = float("-inf")
    for scale in scales:
        new_width, new_height = (
            int(binary_template.shape[1] * scale),
            int(binary_template.shape[0] * scale),
        )
        binary_template_resized = cv2.resize(
            binary_template,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        for angle in rotations:
            M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), angle, 1)
            binary_template_rotated = cv2.warpAffine(
                binary_template_resized,
                M,
                (new_width, new_height),
            )
            matches = cv2.matchTemplate(
                image,
                binary_template_rotated,
                cv2.TM_CCORR_NORMED,
            )
            _, score, _, _ = cv2.minMaxLoc(matches)
            if score > max_score:
                max_score = score
                max_pred = row.label

    return max_score, max_pred


def template_match_classify(
    image: np.ndarray,
    template_images_dir: str,
    image_metadata: pd.DataFrame,
    scales: np.ndarray,
    rotations: np.ndarray,
    image_suffix: str = ".png",
) -> Dict[str, float]:
    """
    Uses template matching to recognize hand shapes.

    :param image: The image to recognize hand shapes in.
    :param template_images_dir: The directory containing the binary template images.
    :param image_metadata: The metadata of the binary template images.
    :param scales: The scales to resize the binary template images to.
    :param rotations: The rotations to rotate the binary template images by.
    :param image_suffix: The suffix of the binary template images (.png by default).
    :return: The predicted label.
    """
    max_pred = 1
    max_score = float("-inf")
    for _, row in image_metadata.iterrows():
        binary_template_image = cv2.imread(
            str(Path(template_images_dir).joinpath(f"{row.image_name}{image_suffix}")),
            flags=cv2.IMREAD_GRAYSCALE,
        )
        if (
            binary_template_image.shape[0] > image.shape[0]
            or binary_template_image.shape[1] > image.shape[1]
        ):
            downscale_factor = max(
                binary_template_image.shape[0] / image.shape[0],
                binary_template_image.shape[1] / image.shape[1],
            )
            new_width, new_height = (
                int(binary_template_image.shape[1] / downscale_factor),
                int(binary_template_image.shape[0] / downscale_factor),
            )
            binary_template_image = cv2.resize(
                binary_template_image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        score, pred = template_match(
            row, image, binary_template_image, scales, rotations
        )

        if score > max_score:
            max_score = score
            max_pred = pred

    return {"pred": max_pred, "score": max_score}
