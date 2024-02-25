from a3.utils import crop_video_dimensions

if __name__ == "__main__":
    source_video = "../data/original/commonwealth.mp4"
    width = 700
    height = 500

    crop_video_dimensions(
        width=width,
        height=height,
        source_video=source_video,
        save_path="../data/cropped/commonwealth.mp4",
    )
