from pathlib import Path
from typing import List, Dict
import json


def save_bounding_boxes(
    bounding_boxes: Dict[str, List[Dict[str, int]]],
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(bounding_boxes, open(save_path, "w"))
