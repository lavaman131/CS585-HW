import cv2
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from a2.algorithms.segmentation import find_max_countour

from a2.data.preprocessing import (
    color_model_binary_image_conversion,
    post_process_binary_image,
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--template_labels_file",
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.375,
        help="Gamma value for adjusting the brightness of the captured frames.",
    )

    args = parser.parse_args()

    template_labels_file = Path(args.template_labels_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    template_images_dir = Path(args.template_images_dir)

    image_metadata = pd.read_csv(template_labels_file)

    for _, row in image_metadata.iterrows():
        image = cv2.imread(str(template_images_dir.joinpath(row.image_name)))
        binary_image = color_model_binary_image_conversion(image, args.gamma)
        c = find_max_countour(binary_image)
        binary_image = post_process_binary_image(binary_image, c)
        cv2.imwrite(str(save_dir.joinpath(row.image_name)), binary_image)

    image_metadata.to_csv(save_dir.joinpath("labels.csv"), index=False)


if __name__ == "__main__":
    main()
