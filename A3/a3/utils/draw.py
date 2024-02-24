import cv2
from typing import Dict, List, Tuple
import numpy as np


def draw_object(
    object_dict: Dict[str, int],
    image: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    # draw box
    x = object_dict["x_min"]
    y = object_dict["y_min"]
    width = object_dict["width"]
    height = object_dict["height"]
    image = cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
    return image


def draw_target_object_center(
    save_path: str, video_file: str, obj_centers: List[List[int]]
) -> None:
    count = 0
    cap = cv2.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"avc1"),  # type: ignore
        30,
        (700, 500),
    )
    while ok:
        pos_x, pos_y = obj_centers[count]
        count += 1
        image = cv2.resize(image, (700, 500))
        image = cv2.circle(image, (pos_x, pos_y), 1, (0, 0, 255), 2)
        vidwrite.write(image)
        ok, image = cap.read()
    vidwrite.release()


def draw_objects_in_video(
    save_path: str, video_file: str, frame_dict: Dict[str, List[Dict[str, int]]]
) -> None:
    count = 0
    cap = cv2.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"avc1"),  # type: ignore
        30,
        (700, 500),
    )
    while ok:
        image = cv2.resize(image, (700, 500))
        obj_list = frame_dict[str(count)]
        for obj in obj_list:
            image = draw_object(obj, image)
        vidwrite.write(image)
        count += 1
        ok, image = cap.read()
    vidwrite.release()
