from pathlib import Path
from typing import List
import json


def save_target_object_centers(
    object_centers: List[List[int]],
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    object_centers_dict = {"obj": object_centers}
    json.dump(object_centers_dict, open(save_path, "w"))
