import json
from typing import Dict


def load_obj_each_frame(data_file: str) -> Dict[str, list]:
    with open(data_file, "r") as file:
        frame_dict = json.load(file)
    return frame_dict
