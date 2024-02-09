import cv2
from a2.data.preprocessing import color_model_binary_image_conversion
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="Path to the template image labels .csv file.",
    )
    parser.add_argument(
        "--template_images_dir",
        type=str,
        required=True,
        help="Path to the directory containing template images.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where template images will be saved.",
    )

    args = parser.parse_args()

    labels_file = Path(args.labels_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    template_images_dir = Path(args.template_images_dir)

    image_metadata = pd.read_csv(labels_file)

    for _, row in image_metadata.iterrows():
        image = cv2.imread(str(template_images_dir.joinpath(row.image_name)))
        binary_image = color_model_binary_image_conversion(image)
        cv2.imwrite(str(save_dir.joinpath(f"{row.label}.png")), binary_image)


if __name__ == "__main__":
    main()
