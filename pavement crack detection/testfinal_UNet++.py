import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os
import glob
from PIL import Image
import numpy as np
from segmentation_models_pytorch import UnetPlusPlus

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Custom Dataset for Testing with Crack Type Handling
class SegmentationTestDataset(Dataset):
    def __init__(self, base_dir, transform_img=None, transform_mask=None):
        self.image_paths = []
        self.mask_paths = []

        crack_types = os.listdir(base_dir)  # Crack type folders

        for crack_type in crack_types:
            img_dir = os.path.join(base_dir, crack_type, "images")
            mask_dir = os.path.join(base_dir, crack_type, "masks")

            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
                mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

                self.image_paths.extend(img_files)
                self.mask_paths.extend(mask_files)

        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# Paths for datasets
base_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\segmentation_data"
train_base_path = os.path.join(base_path, "train")
valid_base_path = os.path.join(base_path, "validation")
test_base_path = os.path.join(base_path, "test")

# Load datasets
train_dataset = SegmentationTestDataset(train_base_path, transform_img, transform_mask)
valid_dataset = SegmentationTestDataset(valid_base_path, transform_img, transform_mask)
test_dataset = SegmentationTestDataset(test_base_path, transform_img, transform_mask)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Load UNet++ model
model = UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=7  # 7 crack classes
)

# Load trained weights
model_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\UNetPlusPlus_model\\UNetPlusPlus_trained_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Compute IoU and Dice Score for multi-class segmentation
def compute_iou_dice(y_true, y_pred, num_classes=7):
    iou_scores = []
    dice_scores = []

    for cls in range(num_classes):
        true_cls = (y_true == cls).astype(np.uint8)
        pred_cls = (y_pred == cls).astype(np.uint8)

        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()

        iou = intersection / (union + 1e-7)  # Avoid division by zero
        dice = (2 * intersection) / (true_cls.sum() + pred_cls.sum() + 1e-7)

        iou_scores.append(iou)
        dice_scores.append(dice)

    return np.mean(iou_scores), np.mean(dice_scores)

# Evaluate Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def evaluate_model(model, data_loader, num_classes=7, name="Dataset"):
    model.eval()
    iou_list, dice_list = [], []
    acc_list, prec_list, rec_list, f1_list, mcc_list = [], [], [], [], []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = torch.softmax(model(images), dim=1)
            preds = torch.argmax(outputs, dim=1)

            masks_np = masks.cpu().numpy().astype(np.uint8)
            preds_np = preds.cpu().numpy().astype(np.uint8)

            # Compute IoU and Dice for this batch
            iou, dice = compute_iou_dice(masks_np, preds_np, num_classes)
            iou_list.append(iou)
            dice_list.append(dice)

            # Flatten batch and compute classification metrics
            y_true_batch = masks_np.flatten()
            y_pred_batch = preds_np.flatten()

            acc_list.append(accuracy_score(y_true_batch, y_pred_batch))
            prec_list.append(precision_score(y_true_batch, y_pred_batch, average='weighted', zero_division=1))
            rec_list.append(recall_score(y_true_batch, y_pred_batch, average='weighted', zero_division=1))
            f1_list.append(f1_score(y_true_batch, y_pred_batch, average='weighted', zero_division=1))
            mcc_list.append(matthews_corrcoef(y_true_batch, y_pred_batch))

            # Optional: free memory
            del images, masks, outputs, preds, masks_np, preds_np
            torch.cuda.empty_cache()

    # Print average metrics
    print(f"\n📊 **{name} Set Evaluation Scores:** 📊")
    print(f"📌 Mean IoU: {np.mean(iou_list) * 100:.2f}%")
    print(f"📌 Mean Dice Score: {np.mean(dice_list) * 100:.2f}%")
    print(f"🎯 Accuracy: {np.mean(acc_list) * 100:.2f}%")
    print(f"📌 Precision: {np.mean(prec_list) * 100:.2f}%")
    print(f"📌 Recall: {np.mean(rec_list) * 100:.2f}%")
    print(f"📌 F1-Score: {np.mean(f1_list) * 100:.2f}%")
    print(f"📎 MCC: {np.mean(mcc_list):.4f}")

if __name__ == '__main__':
    evaluate_model(model, train_loader, name="Train")
    evaluate_model(model, valid_loader, name="Validation")
    evaluate_model(model, test_loader, name="Test")
