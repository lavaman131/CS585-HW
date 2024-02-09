import cv2
from pathlib import Path
from a2.algorithms.segmentation import segment_image
from a2.data.preprocessing import color_model_binary_image_conversion

global LINE_THICKNESS

LINE_THICKNESS = 3


# def capture_video(
#     rectangle_width: int = 640,
#     rectangle_height: int = 790,
#     width: int = 640,
#     height: int = 480,
# ) -> None:
#     """
#     Captures video from the camera and displays it in a window. Press 'q' to quit.
#     :param width: The width of the window
#     :param height: The height of the window
#     :return: None
#     """
#     cap = cv2.VideoCapture(0)
#     cv2.resizeWindow("Window", width, height)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         # if frame is read correctly ret is True
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#         # Our operations on the frame come here
#         cv2.flip(frame, 1, frame)
#         frame_height, frame_width = frame.shape[:2]
#         center = (frame_width // 2, frame_height // 2)
#         offset_x = rectangle_width // 2
#         offset_y = rectangle_height // 2
#         top_left = (center[0] - offset_x, center[1] - offset_y)
#         bottom_right = (center[0] + offset_x, center[1] + offset_y)
#         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
#         # only look at frame within the rectangle
#         region_of_interest = frame[
#             top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
#         ]
#         # Convert frame to binary image
#         binary_image = rgb_to_binary_image(region_of_interest)

#         # Display the resulting frame
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()


def save_segmented_frames(
    save_dir: str,
    rectangle_width: int = 640,
    rectangle_height: int = 790,
    gamma: float = 0.375,
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    num_frames_to_save: int = 5,
    start_delay_seconds: int = 3,
) -> None:
    """
    Captures video frames and saves them as binary images.

    :param save_dir: Directory where frames will be saved.
    :param rectangle_width: Width of the rectangle to capture frames from.
    :param rectangle_height: Height of the rectangle to capture frames from.
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

    cap = cv2.VideoCapture(camera_id)
    cv2.resizeWindow("Window", width, height)

    print("Starting video capture in 3 seconds...")
    sleep_frames = fps * start_delay_seconds
    max_frames = sleep_frames + num_frames_to_save
    frame_number = 0
    image_number = 0

    try:
        while cap.isOpened() and frame_number < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame_height, frame_width = frame.shape[:2]
            center = (frame_width // 2, frame_height // 2)
            offset_x = rectangle_width // 2
            offset_y = rectangle_height // 2
            top_left = (center[0] - offset_x, center[1] - offset_y)
            bottom_right = (center[0] + offset_x, center[1] + offset_y)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), LINE_THICKNESS)

            if frame_number >= sleep_frames:
                if frame_number == sleep_frames:
                    print("Starting video capture...")

                # only look at frame within the rectangle
                region_of_interest = frame[
                    top_left[1] + LINE_THICKNESS : bottom_right[1] - LINE_THICKNESS,
                    top_left[0] + LINE_THICKNESS : bottom_right[0] - LINE_THICKNESS,
                ]

                cropped_image = region_of_interest.copy()
                binary_image = color_model_binary_image_conversion(cropped_image)
                c = segment_image(cropped_image, gamma)

                cv2.drawContours(
                    region_of_interest, [c], -1, (0, 255, 0), LINE_THICKNESS
                )
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
        # Ensure resources are released even in case of an error
        cap.release()
        cv2.destroyAllWindows()
