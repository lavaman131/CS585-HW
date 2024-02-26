import json
from a3.algorithms.object_detection import BoundingBoxMatcher
from a3.utils.draw.object_detection import draw_bounding_boxes_in_video
from a3.utils.io import load_obj_each_frame
from a3.utils.draw.object_tracking import draw_target_object_tracks
from a3.utils.io.object_tracking import save_target_object_centers
from a3.utils.io.object_detection import save_bounding_boxes
from a3.algorithms.object_tracking import AlphaBetaFilter2D
import numpy as np

if __name__ == "__main__":
    frame_dict = load_obj_each_frame("./data/cropped/object_to_track.json")
    fps = 30
    dt = 1.0 / fps

    coords = np.array(frame_dict["obj"])

    alpha_beta_filter_2d = AlphaBetaFilter2D(
        alpha=0.25,
        beta=0.0025,
        x_0=312,
        y_0=228,
        v_x_0=0.0,
        v_y_0=0.0,
        dt=dt,
    )

    corrected_measurements = alpha_beta_filter_2d.predict(coords)

    save_target_object_centers(
        object_centers=corrected_measurements.tolist(),
        save_path="./data/submission/part_1_object_tracking.json",
    )

    draw_target_object_tracks(
        width=700,
        height=500,
        object_centers=corrected_measurements,
        source_video="./data/cropped/commonwealth.mp4",
        save_path="./data/submission/part_1_demo.mp4",
    )

    bounding_boxes = load_obj_each_frame("./data/cropped/frame_dict.json")

    matcher = BoundingBoxMatcher(
        bounding_boxes=bounding_boxes,
        max_distance_threshold=1.0,
        max_frame_skipped=fps,
        fps=fps,
    )

    bounding_boxes = matcher.fit()

    save_bounding_boxes(
        bounding_boxes=bounding_boxes,
        save_path="./data/submission/part_2_frame_dict.json",
    )

    with open("./data/submission/part_2_frame_dict.json") as f:
        bounding_boxes = json.load(f)

    unique_ids = np.unique(
        [bbox["id"] for frame in bounding_boxes for bbox in bounding_boxes[frame]]
    )

    print(unique_ids)

    draw_bounding_boxes_in_video(
        width=700,
        height=500,
        bounding_boxes=bounding_boxes,
        source_video="./data/cropped/commonwealth.mp4",
        save_path="./data/submission/part_2_demo.mp4",
    )
