import torch
import fcn_model
import fcn_dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from typing import Tuple
from pathlib import Path
from torch.optim import AdamW
from lightning import seed_everything

seed_everything(42)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_weights_path = Path(
    "/projectnb/ivc-ml/alavaee/model_weights/vgg16-397923af.pth"
)

# Define the model
num_classes = 32
model = fcn_model.FCN8s(num_classes, pretrain_weights_path).to(device)

# Define the dataset and dataloader
root = "/projectnb/ivc-ml/alavaee/data/CS585/CamVid"
images_dir_train = "train/"
labels_dir_train = "train_labels/"
class_dict_path = "class_dict.csv"
resolution = [384, 512]
batch_size = 16
num_epochs = 200
lr = 1e-4

camvid_dataset_train = fcn_dataset.CamVidDataset(
    root=root,
    images_dir=images_dir_train,
    labels_dir=labels_dir_train,
    class_dict_path=class_dict_path,
    resolution=resolution,
    crop=True,
)
dataloader_train = DataLoader(
    camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4
)

images_dir_val = "val/"
labels_dir_val = "val_labels/"
camvid_dataset_val = fcn_dataset.CamVidDataset(
    root=root,
    images_dir=images_dir_val,
    labels_dir=labels_dir_val,
    class_dict_path=class_dict_path,
    resolution=resolution,
    crop=False,
)
dataloader_val = DataLoader(
    camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4, drop_last=False
)

images_dir_test = "test/"
labels_dir_test = "test_labels/"
camvid_dataset_test = fcn_dataset.CamVidDataset(
    root=root,
    images_dir=images_dir_test,
    labels_dir=labels_dir_test,
    class_dict_path=class_dict_path,
    resolution=resolution,
    crop=False,
)
dataloader_test = DataLoader(
    camvid_dataset_test, batch_size=1, shuffle=False, num_workers=4, drop_last=False
)


# Define the loss function and optimizer
def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(outputs, labels, reduction="mean")


def calculate_pixel_accuracy(confusion_matrix: torch.Tensor) -> float:
    correct = confusion_matrix.diag().sum().item()
    total = confusion_matrix.sum().item()
    return correct / total


def calculate_iou(confusion_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    intersection = confusion_matrix.diag()
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection
    return intersection, union


def calculate_mean_iou(confusion_matrix: torch.Tensor) -> float:
    intersection, union = calculate_iou(confusion_matrix)
    iou = intersection / union
    return torch.nanmean(iou).item()


def calculate_frequency_weighted_iou(
    confusion_matrix: torch.Tensor, epsilon: float = 1e-8
) -> float:
    intersection, union = calculate_iou(confusion_matrix)
    iou = intersection / (union + epsilon)
    frequency = confusion_matrix.sum(dim=1) / confusion_matrix.sum()
    return (frequency * iou).sum().item()


optimizer = AdamW(model.parameters(), lr=lr)


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_pred: bool = False,
):
    model.eval()
    loss_list = []
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    if save_pred:
        pred_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # shape: (batch_size, num_classes, H, W)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, dim=1)  # shape: (batch_size, H, W)
            indices = (labels.view(-1), predicted.view(-1))
            values = torch.ones_like(labels.view(-1), device=device)
            confusion_matrix.index_put_(indices, values, accumulate=True)
            if save_pred:
                pred_list.append(predicted.cpu().numpy())
        pixel_acc = calculate_pixel_accuracy(confusion_matrix)
        mean_iou = calculate_mean_iou(confusion_matrix)
        freq_iou = calculate_frequency_weighted_iou(confusion_matrix)
        loss = sum(loss_list) / len(loss_list)
        print(
            "Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Loss: {:.4f}".format(
                pixel_acc, mean_iou, freq_iou, loss
            )
        )

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save("test_pred.npy", pred_list)
    model.train()


def visualize_model(model, dataloader, device):
    log_dir = "vis/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cls_dict = dataloader.dataset.class_dict.copy()
    cls_list = [cls_dict[i] for i in range(len(cls_dict))]
    model.eval()
    with torch.no_grad():
        for ind, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            images_vis = fcn_dataset.rev_normalize(images)
            # Save the images and labels
            img = images_vis[0].permute(1, 2, 0).cpu().numpy()
            img = img.astype("uint8")
            label = labels[0].cpu().numpy()
            pred = predicted[0].cpu().numpy()

            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            pred_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for j in range(len(cls_list)):
                mask = label == j
                label_img[mask] = cls_list[j][0]
                mask = pred == j
                pred_img[mask] = cls_list[j][0]
            # horizontally concatenate the image, label, and prediction, and save the visualization
            vis_img = np.concatenate([img, label_img, pred_img], axis=1)
            vis_img = Image.fromarray(vis_img)
            vis_img.save(os.path.join(log_dir, "img_{:04d}.png".format(ind)))

    model.train()


# Train the model
loss_list = []
scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader_train):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # zero grad
        optimizer.zero_grad()

        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            loss = loss_fn(outputs, labels)

        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        loss_list.append(loss.item())

        if (i + 1) % 10 == 0:
            loss = sum(loss_list) / len(loss_list)
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    len(dataloader_train),
                    loss,
                )
            )
            loss_list = []
    # eval the model
    eval_model(model, dataloader_val, device)

print("=" * 20)
print("Finished Training, evaluating the model on the test set")
eval_model(model, dataloader_test, device, save_pred=True)

print("=" * 20)
print(
    "Visualizing the model on the test set, the results will be saved in the vis/ directory"
)
visualize_model(model, dataloader_test, device)
