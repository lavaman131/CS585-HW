from a3.utils import load_obj_each_frame
from a3.utils.draw import draw_objects_in_video, draw_target_object_center

if __name__ == "__main__":
    video_file = "../data/commonwealth.mp4"

    frame_dict = load_obj_each_frame("../data/object_to_track.json")
    draw_target_object_center("../data/part_1_demo.mp4", video_file, frame_dict["obj"])

    frame_dict = load_obj_each_frame("../data/frame_dict.json")
    draw_objects_in_video("../data/part_2_demo.mp4", video_file, frame_dict)
