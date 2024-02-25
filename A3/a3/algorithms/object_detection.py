from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.linalg as LA
from typing import Dict, List, Tuple


class BoundingBoxMatcher:
    def __init__(
        self,
        bounding_boxes: Dict[str, List[Dict[str, int]]],
        max_distance_threshold: float,
        max_frame_skipped: int,
    ) -> None:
        self.bounding_boxes = bounding_boxes
        self.max_distance_threshold = max_distance_threshold
        self.max_frame_skipped = max_frame_skipped
        self.tracks = []
        self.track_id = 0

    def _add_new_track(self, bbox: Dict[str, int]) -> None:
        track = {
            "id": self.track_id,
            **bbox,
            "skipped_frames": 0,
        }
        self.tracks.append(track)
        self.track_id += 1

    def postprocess(self) -> None:
        for frame in self.bounding_boxes:
            for bbox in self.bounding_boxes[frame]:
                if "id" not in bbox:
                    bbox["id"] = self.track_id
                    self.track_id += 1

    def fit(self) -> Dict[str, List[Dict[str, int]]]:
        for frame in range(len(self.bounding_boxes)):
            detections = self.bounding_boxes[str(frame)]
            self.update(detections)

        self.postprocess()

        return self.bounding_boxes

    def _init_tracks(self, detections: List[Dict[str, int]]) -> None:
        for i, det in enumerate(detections):
            self._add_new_track(det)
            detections[i]["id"] = self.tracks[i]["id"]

    def update(self, detections: List[Dict[str, int]]) -> None:
        if not self.tracks:
            self._init_tracks(detections)
            return

        # Create the cost matrix
        cost_matrix = self._calculate_cost_matrix(detections)

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update tracks based on the assignment
        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_distance_threshold:
                detections[c]["id"] = self.tracks[r]["id"]
                self.tracks[r]["skipped_frames"] = 0
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # Increment skipped frames for unmatched tracks
        for i in range(len(self.tracks)):
            if i not in assigned_tracks:
                self.tracks[i]["skipped_frames"] += 1

        # Add new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in assigned_detections:
                self._add_new_track(detections[i])

        # Remove tracks that have exceeded the max_frame_skipped threshold
        self.tracks = [
            track
            for track in self.tracks
            if track["skipped_frames"] <= self.max_frame_skipped
        ]

    def _hungarian_algorithm(
        self, cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # implement the Hungarian algorithm from scratch
        pass

    def _calculate_cost_matrix(self, detections: List[Dict[str, int]]) -> np.ndarray:
        num_tracks = len(self.tracks)
        num_detections = len(detections)
        cost_matrix = np.zeros((num_tracks, num_detections))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                track_centroid = np.array(
                    [
                        track["x_min"] + track["width"] / 2.0,
                        track["y_min"] + track["height"] / 2.0,
                    ]
                )
                det_centroid = np.array(
                    [
                        det["x_min"] + det["width"] / 2.0,
                        det["y_min"] + det["height"] / 2.0,
                    ]
                )
                cost_matrix[i, j] = LA.norm(track_centroid - det_centroid)

        max_distance = np.max(cost_matrix)
        normalized_cost_matrix = cost_matrix / max_distance

        return normalized_cost_matrix
