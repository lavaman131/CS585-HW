import cv2
from a2.algorithms.segmentation import segment_image
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


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
        c = segment_image(image, args.gamma)
        binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(binary_image, [c], (255, 255, 255))
        cv2.imwrite(str(save_dir.joinpath(row.image_name)), binary_image)

    image_metadata.to_csv(save_dir.joinpath("labels.csv"), index=False)


if __name__ == "__main__":
    main()
