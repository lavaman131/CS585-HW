from pathlib import Path
import cv2
import cv2.typing
from typing import List, Tuple

import numpy as np

from a3 import MISSING_VALUE


def draw_target_object_center(
    image: cv2.typing.MatLike,
    pos_x: int,
    pos_y: int,
    radius: int = 1,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> cv2.typing.MatLike:
    image = cv2.circle(image, (pos_x, pos_y), radius, color, thickness)
    return image


def draw_target_object_centers(
    width: int,
    height: int,
    object_centers: List[List[int]],
    source_video: str,
    save_path: str,
    codec: str = "avc1",
    fps: int = 30,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    cap = cv2.VideoCapture(source_video)
    ok, image = cap.read()
    vidwrite = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*codec),  # type: ignore
        fps,
        (width, height),
    )
    while ok:
        pos_x, pos_y = object_centers[count]
        count += 1
        image = cv2.resize(image, (width, height))
        image = draw_target_object_center(image, pos_x, pos_y)
        vidwrite.write(image)
        ok, image = cap.read()
    vidwrite.release()


def draw_target_object_track(
    image: cv2.typing.MatLike,
    coords: cv2.typing.MatLike,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> cv2.typing.MatLike:
    return cv2.polylines(image, [coords], False, color, thickness)


def draw_target_object_tracks(
    width: int,
    height: int,
    object_centers: cv2.typing.MatLike,
    source_video: str,
    save_path: str,
    codec: str = "avc1",
    fps: int = 30,
) -> None:
    assert len(object_centers) > 0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    cap = cv2.VideoCapture(source_video)
    ok, image = cap.read()
    vidwrite = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*codec),  # type: ignore
        fps,
        (width, height),
    )
    curr_track = []
    while ok and count < len(object_centers):
        pos_x, pos_y = object_centers[count]
        image = cv2.resize(image, (width, height))
        if pos_x != MISSING_VALUE and pos_y != MISSING_VALUE:
            curr_track.append((pos_x, pos_y))
        if len(curr_track) > 1:
            image = draw_target_object_track(image, np.array(curr_track))
        vidwrite.write(image)
        count += 1
        ok, image = cap.read()
    vidwrite.release()
