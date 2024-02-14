import cv2
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from a2.algorithms.segmentation import find_max_countour
import numpy as np
from a2.data.preprocessing import (
    color_model_binary_image_conversion,
    post_process_binary_image,
)


def save_binary_image(
    image: np.ndarray,
    image_name: str,
    image_suffix: str,
    gamma: float,
    post_process: bool,
    save_dir: Path,
) -> None:
    """
    Saves the binary image of the region of interest.
    :param image: The image to save the binary image of.
    :param image_name: Name of the image.
    :param image_suffix: Suffix of the image.
    :param gamma: Gamma value for adjusting the brightness of the captured frames.
    :param post_process: Whether to post process the binary image.
    :param save_dir: Directory where the binary images will be saved.
    """
    binary_image = color_model_binary_image_conversion(image, gamma)
    c = find_max_countour(binary_image)
    if post_process:
        binary_image = post_process_binary_image(binary_image, c)
    cv2.imwrite(str(save_dir.joinpath(f"{image_name}{image_suffix}")), binary_image)


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
    parser.add_argument(
        "--post_process",
        action="store_true",
        type=bool,
        default=False,
        help="Post process binary images.",
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=".png",
        help="Suffix of the template images. Default is .png.",
    )

    args = parser.parse_args()

    template_labels_file = Path(args.template_labels_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    template_images_dir = Path(args.template_images_dir)

    image_metadata = pd.read_csv(template_labels_file)
    new_image_metadata = []

    for _, row in image_metadata.iterrows():
        fname = str(
            template_images_dir.joinpath(f"{row.image_name}{args.image_suffix}")
        )
        image = cv2.imread(fname)
        save_binary_image(
            image,
            row.image_name,
            args.image_suffix,
            args.gamma,
            args.post_process,
            save_dir,
        )
        new_image_metadata.append(row.tolist())
        flipped_image_name = f"{row.image_name}_flipped"
        save_binary_image(
            cv2.flip(image, 1),
            flipped_image_name,
            args.image_suffix,
            args.gamma,
            args.post_process,
            save_dir,
        )
        new_image_metadata.append([flipped_image_name, row.label])

    new_image_metadata = pd.DataFrame(
        new_image_metadata, columns=["image_name", "label"]
    )

    new_image_metadata.to_csv(save_dir.joinpath("labels.csv"), index=False)


if __name__ == "__main__":
    main()
