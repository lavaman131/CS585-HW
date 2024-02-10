import cv2
import pandas as pd
from pathlib import Path
from a2.algorithms.classification import template_match_classify
from a2.algorithms.segmentation import segment_image
from a2.data.preprocessing import color_model_binary_image_conversion
import csv
from colorama import Fore

global LINE_THICKNESS

LINE_THICKNESS = 3


def predict(
    ground_truth_label: int,
    template_labels_file: str,
    template_images_dir: str,
    save_dir: str,
    roi_width: int = 640,
    roi_height: int = 790,
    gamma: float = 0.375,
    camera_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    num_frames_to_save: int = 5,
    start_delay_seconds: int = 3,
) -> None:
    """
    Captures video frames and saves binary and RGB images of the region of interest along with the predicted label.
    :param ground_truth_label: ground truth label for the frames.
    :param template_labels_file: Path to the template image labels .csv file.
    :param template_images_dir: Directory containing the binary template images.
    :param save_dir: Directory where frames will be saved.
    :param roi_width: Width of the rectangular region of interest to capture frames from.
    :param roi_height: Height of the rectangular region of interest to capture frames from.
    :param gamma: Gamma value for adjusting the brightness of the captured frames.
    :param camera_id: ID of the camera to capture frames from.
    :param width: Width of the captured frames.
    :param height: Height of the captured frames.
    :param fps: Frames per second for video capture.
    :param num_frames_to_save: Number of frames to save.
    :param start_delay_seconds: Delay before starting capture, in seconds.
    :param binary_threshold: Threshold for converting frames to binary images.
    :return: None
    """
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
            if not ret:
                print(Fore.RED + "Can't receive frame (stream end?). Exiting ...")
                break
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
                binary_image = color_model_binary_image_conversion(cropped_image)
                c = segment_image(cropped_image, gamma)

                result = template_match_classify(
                    binary_image, template_images_dir, image_metadata
                )

                pred, score = result["pred"], result["score"]

                stats_writer.writerow([frame_number, pred, ground_truth_label, score])

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
                    region_of_interest, [c], -1, (0, 255, 0), LINE_THICKNESS
                )
                cv2.fillPoly(binary_image, [c], (255, 255, 255))
                # save without rectangle
                cv2.imwrite(
                    str(save_path / f"frame_{image_number}_rgb.png"), region_of_interest
                )

                cv2.imwrite(
                    str(save_path / f"frame_{image_number}_binary.png"), binary_image
                )

                image_number += 1

            cv2.imshow("frame", frame)
            frame_number += 1

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        stats.close()
        # Ensure resources are released even in case of an error
        cap.release()
        cv2.destroyAllWindows()
