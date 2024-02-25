from pathlib import Path
import cv2
from typing import Dict, List, Tuple
import cv2.typing


def draw_bounding_box(
    bounding_box: Dict[str, int],
    image: cv2.typing.MatLike,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> cv2.typing.MatLike:
    # draw box
    x = bounding_box["x_min"]
    y = bounding_box["y_min"]
    width = bounding_box["width"]
    height = bounding_box["height"]
    image = cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
    return image


def annotate_bounding_box(
    bounding_box: Dict[str, int],
    image: cv2.typing.MatLike,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> cv2.typing.MatLike:
    x = bounding_box["x_min"]
    y = bounding_box["y_min"]
    width = bounding_box["width"]
    height = bounding_box["height"]
    x_mid = x + width // 2
    y_mid = y + height // 2
    text = str(bounding_box["id"])

    image = cv2.putText(
        image,
        text,
        (x_mid, y_mid),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )

    return image


def draw_bounding_boxes_in_video(
    width: int,
    height: int,
    bounding_boxes: Dict[str, List[Dict[str, int]]],
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
        image = cv2.resize(image, (width, height))
        bboxes = bounding_boxes[str(count)]
        for bbox in bboxes:
            image = draw_bounding_box(bbox, image)
            image = annotate_bounding_box(bbox, image)
        vidwrite.write(image)
        count += 1
        ok, image = cap.read()
    vidwrite.release()
