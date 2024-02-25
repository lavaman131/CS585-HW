import json
from typing import Dict
from pathlib import Path
import cv2


def load_obj_each_frame(data_file: str) -> Dict[str, list]:
    with open(data_file, "r") as file:
        frame_dict = json.load(file)
    return frame_dict


def crop_video_dimensions(
    width: int,
    height: int,
    source_video: str,
    save_path: str,
    codec: str = "avc1",
    fps: int = 30,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
        vidwrite.write(image)
        ok, image = cap.read()
    vidwrite.release()
