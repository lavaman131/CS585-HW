import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from torchvision import models


class FCN8s(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrain_weights_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()

        # Load the pretrained VGG-16 and use its features
        if pretrain_weights_path:
            vgg16 = models.vgg16(weights=None)
            vgg16.load_state_dict(torch.load(pretrain_weights_path))
        else:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
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
        x1 = self.features_block1(x)
        x2 = self.features_block2(x1)
        x3 = self.features_block3(x2)
        x4 = self.features_block4(x3)
        x5 = self.features_block5(x4)

        score_fr = self.classifier(x5)
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(x4)
        upscore2c = FCN8s.crop(upscore2, score_pool4.size())
        fuse_pool4 = upscore2c + score_pool4
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(x3)
        score_pool3c = FCN8s.crop(score_pool3, upscore_pool4.size())
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore_final = self.upscore_final(fuse_pool3)
        upscore_finalc = FCN8s.crop(upscore_final, x.size())

        return upscore_finalc

    @staticmethod
    def crop(input_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        _, _, input_height, input_width = input_tensor.size()
        _, _, target_height, target_width = target_shape
        offset_y = (input_height - target_height) // 2
        offset_x = (input_width - target_width) // 2
        return input_tensor[
            ...,
            offset_y : offset_y + target_height,
            offset_x : offset_x + target_width,
        ]


if __name__ == "__main__":
    num_classes = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_weights_path = Path(
        "/projectnb/ivc-ml/alavaee/model_weights/vgg16-397923af.pth"
    )
    model = FCN8s(num_classes, pretrain_weights_path).to(device)
    x = torch.randn(1, 3, 384, 512).to(device)
    y = model(x)
    assert y.shape == torch.Size([1, num_classes, 384, 512])
