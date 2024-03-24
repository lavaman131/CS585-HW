import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.io as vio
from torchvision.transforms import v2
import numpy as np
from typing import List, Tuple, Dict, TypeAlias

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
rev_normalize = v2.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def paired_crop_and_resize(
    image: torch.Tensor, label: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Random crop
    i, j, h, w = v2.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1, 1))  # type: ignore
    image = v2.functional.resized_crop(
        image, i, j, h, w, size, v2.InterpolationMode.BILINEAR
    )
    label = v2.functional.resized_crop(
        label, i, j, h, w, size, v2.InterpolationMode.NEAREST
    )

    return image, label


def paired_resize(
    image: torch.Tensor, label: torch.Tensor, size: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = v2.functional.resize(image, size, v2.InterpolationMode.BILINEAR)
    label = v2.functional.resize(label, size, v2.InterpolationMode.NEAREST)

    return image, label


class CamVidDataset(Dataset):
    ClassDict: TypeAlias = Dict[int, Tuple[Tuple[int, int, int], str]]

    def __init__(
        self,
        root: str,
        images_dir: str,
        labels_dir: str,
        class_dict_path: str,
        resolution: List[int],
        crop: bool = False,
    ) -> None:
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.resolution = resolution
        self.crop = crop
        self.class_dict = self.parse_class_dict(os.path.join(root, class_dict_path))
        self.rgb_to_class_id_dict = {
            rgb: class_id for class_id, (rgb, _) in self.class_dict.items()
        }
        self.images = [
            os.path.join(root, images_dir, img)
            for img in sorted(os.listdir(os.path.join(root, images_dir)))
        ]
        self.labels = [
            os.path.join(root, labels_dir, lbl)
            for lbl in sorted(os.listdir(os.path.join(root, labels_dir)))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        label_path = self.labels[idx]
        image = vio.read_image(image_path, mode=vio.ImageReadMode.RGB).float()
        label = vio.read_image(label_path, mode=vio.ImageReadMode.RGB)
        if self.crop:
            image, label = paired_crop_and_resize(image, label, self.resolution)
        else:
            image, label = paired_resize(image, label, self.resolution)

        label = self.rgb_to_class_id(label).long()
        # normalize image
        image = normalize(image)

        return image, label

    def parse_class_dict(self, class_dict_path: str) -> ClassDict:
        # return a dictionary that maps class id (0-31) to a tuple ((R,G,B), class_name)
        class_dict = {}
        reader = csv.reader(open(class_dict_path, "r"), delimiter=",")
        # skip the header
        next(reader)
        for i, row in enumerate(reader):
            class_name = row[0]
            r, g, b = row[1:4]
            class_dict[i] = (
                (int(r), int(g), int(b)),
                class_name,
            )
        return class_dict

    def rgb_to_class_id(self, label_img: torch.Tensor) -> torch.Tensor:
        # Convert an RGB label image to a class ID image (3, H, W) -> (H, W)
        label = torch.zeros(label_img.shape[1:])
        for i in range(label_img.shape[1]):
            for j in range(label_img.shape[2]):
                rgb = tuple(label_img[:, i, j].tolist())
                label[i, j] = self.rgb_to_class_id_dict[rgb]
        return label
