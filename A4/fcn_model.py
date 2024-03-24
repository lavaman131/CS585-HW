import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
import torchvision.models as models
from torchvision.transforms import v2


class FCN8s(nn.Module):
    def __init__(
        self, num_classes: int, vgg16_weights_path: Union[str, Path, None] = None
    ):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        if vgg16_weights_path:
            vgg16 = models.vgg16(weights=None)
            vgg16.load_state_dict(torch.load(vgg16_weights_path))
        else:
            vgg16 = models.vgg16(weights=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore_final = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, bias=False
        )

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        block1 = self.features_block1(x)
        block2 = self.features_block2(block1)
        block3 = self.features_block3(block2)
        block4 = self.features_block4(block3)
        block5 = self.features_block5(block4)

        # Classifier
        score = self.classifier(block5)

        # Decoder
        upscore2 = self.upscore2(score)
        score_pool4 = self.score_pool4(block4)
        fuse_pool4 = FCN8s.crop_and_add(score_pool4, upscore2)
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(block3)
        fuse_pool3 = FCN8s.crop_and_add(score_pool3, upscore_pool4)

        upscore_final = self.upscore_final(fuse_pool3)

        return upscore_final

    @staticmethod
    def crop_and_add(
        tensor_to_crop: torch.Tensor, tensor_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Crop tensor_to_crop to match the size of tensor_target and then add them.
        Assumes tensors are in the format [N, C, H, W].
        """
        size_target = tensor_target.size()[2:]
        size_to_crop = tensor_to_crop.size()[2:]

        # Calculate cropping start indices
        start = [(stc - stt) // 2 for stc, stt in zip(size_to_crop, size_target)]

        # Perform cropping
        cropped_tensor = tensor_to_crop[
            :,
            :,
            start[0] : (start[0] + size_target[0]),
            start[1] : (start[1] + size_target[1]),
        ]

        # Add cropped tensor to target tensor
        return cropped_tensor + tensor_target
