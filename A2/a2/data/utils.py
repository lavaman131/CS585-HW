import cv2
import pandas as pd
from pathlib import Path
from a2.algorithms.classification import template_match_classify
from a2.algorithms.segmentation import find_max_countour
from a2.data.preprocessing import color_model_binary_image_conversion
import csv
from colorama import Fore
import numpy as np

global LINE_THICKNESS
LINE_THICKNESS = 3


def predict(
    camera_id: int,
    save_dir: Path,
    num_frames_to_save: int,
    ground_truth_label: int,
    start_delay_seconds: int,
    width: int,
    height: int,
    roi_width: int,
    roi_height: int,
    gamma: float,
    rotations: np.ndarray,
    scales: np.ndarray,
    fps: int,
    template_labels_file: str,
    template_images_dir: str,
) -> None:
    """
    Captures video frames and saves binary and RGB images of the region of interest along with the predicted label.
    :param camera_id: ID of the camera to capture frames from.
    :param save_dir: Directory where frames will be saved.
    :param num_frames_to_save: Number of frames to save.
    :param ground_truth_label: ground truth label for the frames.
    :param start_delay_seconds: Delay before starting capture, in seconds.
    :param width: Width of the captured frames.
    :param height: Height of the captured frames.
    :param roi_width: Width of the rectangular region of interest to capture frames from.
    :param roi_height: Height of the rectangular region of interest to capture frames from.
    :param gamma: Gamma value for adjusting the brightness of the captured frames.
    :param rotations: The rotations to rotate the binary template images by during template matching.
    :param scales: The scales to resize the binary template images to during template matching.
    :param fps: Frames per second for video capture.
    :param template_labels_file: Path to the template image labels .csv file.
    :param template_images_dir: Directory containing the binary template images.
    :return: None
    """
    assert all(s <= 1.0 for s in scales), "All scales must be at most 1."
    ground_truth_label = int(ground_truth_label)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    image_metadata = pd.read_csv(Path(template_labels_file))

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(Fore.YELLOW + "Starting video capture in 3 seconds...")
    sleep_frames = fps * start_delay_seconds
    max_frames = sleep_frames + num_frames_to_save
    frame_number = 0
    image_number = 0

    metadata = {
        "roi_width": roi_width,
        "roi_height": roi_height,
        "gamma": gamma,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames_to_save": num_frames_to_save,
        "start_delay_seconds": start_delay_seconds,
    }

    metadata = pd.DataFrame(
        metadata.items(),
        index=range(len(metadata)),
        columns=["key", "value"],
    )

    metadata.to_csv(save_path.joinpath("metadata.csv"), index=False)

    stats = open(save_path.joinpath("stats.csv"), "w", newline="")
    stats_writer = csv.writer(
        stats, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    stats_writer.writerow(
        [
            "frame_number",
            "predicted_label",
            "ground_truth_label",
            "template_matching_score",
        ]
    )

    try:
        while cap.isOpened() and frame_number < max_frames:
            ret, frame = cap.read()
            # flip frame horizontally
            # frame = cv2.flip(frame, 1)
            if not ret:
                raise ValueError(
                    Fore.RED + "Can't receive frame (stream end?). Exiting ..."
                )
            frame_height, frame_width = frame.shape[:2]
            center = (frame_width // 2, frame_height // 2)
            offset_x = roi_width // 2
            offset_y = roi_height // 2
            top_left = (center[0] - offset_x, center[1] - offset_y)
            bottom_right = (center[0] + offset_x, center[1] + offset_y)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), LINE_THICKNESS)

            if frame_number >= sleep_frames:
                if frame_number == sleep_frames:
                    print(Fore.GREEN + "Starting video capture...")

                # only look at frame within the rectangle
                region_of_interest = frame[
                    top_left[1] + LINE_THICKNESS : bottom_right[1] - LINE_THICKNESS,
                    top_left[0] + LINE_THICKNESS : bottom_right[0] - LINE_THICKNESS,
                ]

                cropped_image = region_of_interest.copy()
                binary_image = color_model_binary_image_conversion(cropped_image, gamma)
                cnts = cv2.findContours(
                    binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]

                result = template_match_classify(
                    binary_image, template_images_dir, image_metadata, scales, rotations
                )

                pred, score = result["pred"], result["score"]

                stats_writer.writerow(
                    [
                        frame_number,
                        pred,
                        "" if ground_truth_label == -1 else ground_truth_label,
                        score,
                    ]
                )

                cv2.putText(
                    region_of_interest,
                    f"Predicted label: {str(pred)}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if ground_truth_label != -1:
                    cv2.putText(
                        region_of_interest,
                        f"Ground truth label: {str(ground_truth_label)}",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.drawContours(
                    region_of_interest, cnts, -1, (0, 255, 0), LINE_THICKNESS
                )

                # save without rectangle
                cv2.imwrite(
                    str(save_path / f"frame_{image_number}_rgb.png"), region_of_interest
                )

                cv2.imwrite(
                    str(save_path / f"frame_{image_number}_binary.png"), binary_image
                )

                image_number += 1

            if cv2.waitKey(1) == ord("q"):
                break

            cv2.imshow("frame", frame)
            frame_number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Ensure resources are released even in case of a NFS I/O error
        try:
            stats.close()
        except TimeoutError:
            stats.close()
