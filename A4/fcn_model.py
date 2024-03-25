import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from torchvision.models import VGG
from typing import Dict


class VGG16(VGG):
    BLOCK_RANGES = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))

    def __init__(
        self, num_classes: int, pretrain_weights_path: Union[str, Path, None]
    ) -> None:
        features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        super().__init__(features, num_classes=num_classes, init_weights=True)

        self.pretrain_weights_path = pretrain_weights_path

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.pretrain_weights_path:
            model_state_dict = self.state_dict()
            pretrained_state_dict = torch.load(self.pretrain_weights_path)
            new_state_dict = {}
            for key in pretrained_state_dict.keys():
                # ignore 1000 classes
                if "classifier.6" in key:
                    continue
                elif "classifier" in key:
                    new_state_dict[key] = pretrained_state_dict[key].view(
                        model_state_dict[key].shape
                    )
                else:
                    new_state_dict[key] = pretrained_state_dict[key]
            self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = {}

        for idx in range(len(VGG16.BLOCK_RANGES)):
            for layer in range(VGG16.BLOCK_RANGES[idx][0], VGG16.BLOCK_RANGES[idx][1]):
                x = self.features[layer](x)  # type: ignore
            output[f"x{(idx + 1)}"] = x

        return output


class FCN8s(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrain_weights_path: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()

        self.vgg16 = VGG16(num_classes, pretrain_weights_path)

        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=4,
            stride=2,
            bias=False,
        )
        self.upscore8 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=16,
            stride=8,
            bias=False,
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=4,
            stride=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.vgg16(x)
        x5 = output["x5"]
        x4 = output["x4"]
        x3 = output["x3"]

        score_fr = self.vgg16.classifier(x5)
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(x4)
        score_pool4c = score_pool4[
            :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
        ]
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(x3)
        score_pool3c = score_pool3[
            :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
        ]
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)
        upscore8c = upscore8[
            :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
        ].contiguous()

        return upscore8c


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
