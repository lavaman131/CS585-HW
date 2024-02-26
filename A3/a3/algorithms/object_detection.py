import numpy as np
import numpy.linalg as LA
from typing import Dict, List
from a3.algorithms.matching import HungarianMatcher
from a3.algorithms.object_tracking import AlphaBetaFilter2D


class Track:
    def __init__(
        self,
        track_id: int,
        bbox: Dict[str, int],
        v_x_0: float = 0.0,
        v_y_0: float = 0.0,
        dt: float = 1.0,
        alpha: float = 0.25,
        beta: float = 0.0025,
    ) -> None:
        """Initializes a track with an alpha-beta filter.

        Parameters
        ----------
        track_id : int
            Unique identifier for the track.
        bbox : Dict[str, int]
            Bounding box of the object.
        v_x_0 : float, optional
            Initial x velocity of the object, by default 0.0
        v_y_0 : float, optional
            Initial y velocity of the object, by default 0.0
        dt : float, optional
            Time step between frames (seconds), by default 1.0
        alpha : float, optional
            Alpha parameter for the alpha-beta filter. A higher alpha value is more sensitive to changes in the measurements, by default 0.25
        beta : float, optional
            Beta parameter for the alpha-beta filter. A higher beta value is more sensitive to changes in the measurements, by default 0.0025
        """
        measurement = (bbox["x_min"], bbox["y_min"])
        self.filter = AlphaBetaFilter2D(
            alpha=alpha,
            beta=beta,
            x_0=measurement[0] + bbox["width"] / 2,
            y_0=measurement[1] + bbox["height"] / 2,
            v_x_0=v_x_0,
            v_y_0=v_y_0,
            dt=dt,
        )
        self.track_id = track_id
        self.update(bbox)

    def update(self, bbox: Dict[str, int]) -> None:
        """Corrects the 2D coordinates of the object using the alpha-beta filter.

        Parameters
        ----------
        bbox : Dict[str, int]
            Bounding box of the object.
        """
        measurement = (bbox["x_min"], bbox["y_min"])
        self.prediction = self.filter(measurement)
        self.skipped_frames = 0


class BoundingBoxMatcher:
    def __init__(
        self,
        bounding_boxes: Dict[str, List[Dict[str, int]]],
        max_distance_threshold: float,
        max_frame_skipped: int,
        fps: int,
    ) -> None:
        """Initializes the bounding box matcher.

        Parameters
        ----------
        bounding_boxes : Dict[str, List[Dict[str, int]]]
            Bounding boxes of the object in each frame.
        max_distance_threshold : float
            Maximum distance threshold for matching tracks with detections.
        max_frame_skipped : int
            Maximum number of frames to skip before a track is removed.
        fps : int
            Frames per second of the video.
        """
        self.bounding_boxes = bounding_boxes
        self.max_distance_threshold = max_distance_threshold
        self.max_frame_skipped = max_frame_skipped
        self.tracks: List[Track] = []
        self.track_id = 0
        self.fps = fps
        self.matcher = HungarianMatcher()

    def _add_new_track(self, bbox: Dict[str, int]) -> None:
        """Adds a new track to the list of tracks.

        Parameters
        ----------
        bbox : Dict[str, int]
            Bounding box of the object.
        """
        dt = 1.0 / self.fps
        track = Track(track_id=self.track_id, bbox=bbox, dt=dt)
        self.tracks.append(track)
        self.track_id += 1

    def fit(self) -> Dict[str, List[Dict[str, int]]]:
        """Fits the bounding box matcher to the data.

        Returns
        -------
        self.bounding_boxes : Dict[str, List[Dict[str, int]]]
            Bounding boxes of the object in each frame.
        """
        for frame in range(len(self.bounding_boxes)):
            detections = self.bounding_boxes[str(frame)]
            self.update(detections)

        return self.bounding_boxes

    def _init_tracks(self, detections: List[Dict[str, int]]) -> None:
        """Initializes the tracks with the first set of detections.

        Parameters
        ----------
        detections : List[Dict[str, int]]
            List of detections in the first frame.
        """
        for i, det in enumerate(detections):
            self._add_new_track(det)
            detections[i]["id"] = self.tracks[i].track_id

    def update(self, detections: List[Dict[str, int]]) -> None:
        """Updates the tracks with the new set of detections.

        Parameters
        ----------
        detections : List[Dict[str, int]]
            List of detections in the current frame.
        """
        if not self.tracks:
            self._init_tracks(detections)
            return

        # Create the cost matrix
        cost_matrix = self._calculate_cost_matrix(detections)

        # Apply the Hungarian algorithm
        row_inds, col_inds = self.matcher(cost_matrix)

        # Update tracks based on the assignment
        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] <= self.max_distance_threshold:
                detections[c]["id"] = self.tracks[r].track_id
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # Increment skipped frames for unmatched tracks
        for i in range(len(self.tracks)):
            if i not in assigned_tracks:
                self.tracks[i].skipped_frames += 1

        # Add new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in assigned_detections:
                self._add_new_track(detections[i])
                detections[i]["id"] = self.tracks[-1].track_id

        # Remove tracks that have exceeded the max_frame_skipped threshold
        self.tracks = [
            track
            for track in self.tracks
            if track.skipped_frames <= self.max_frame_skipped
        ]

    def _calculate_cost_matrix(self, detections: List[Dict[str, int]]) -> np.ndarray:
        """Calculates the cost matrix for the Hungarian algorithm.

        Parameters
        ----------
        detections : List[Dict[str, int]]
            List of detections in the current frame.

        Returns
        -------
        normalized_cost_matrix : np.ndarray
            Cost matrix for the Hungarian algorithm normalized by the maximum distance [0, 1].
        """
        num_tracks = len(self.tracks)
        num_detections = len(detections)
        cost_matrix = np.zeros((num_tracks, num_detections))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                track_centroid = np.array(track.prediction)
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
