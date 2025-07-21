import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from segmentation_models_pytorch import UnetPlusPlus
from PIL import Image
import glob

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

# Crack types
crack_types = [
    "alligator_crack", "block_crack", "longitudinal_crack", "pothole", 
    "sealed_longitudinal_crack", "sealed_transverse_crack", "transverse_crack"
]

# Custom Dataset to load images, masks, and labels
class SegmentationDataset(Dataset):
    def __init__(self, base_dir, crack_types, transform_img=None, transform_mask=None):
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        for idx, crack in enumerate(crack_types):
            image_dir = os.path.join(base_dir, crack, "images")
            mask_dir = os.path.join(base_dir, crack, "masks")
            
            img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
            
            self.image_paths.extend(img_paths)
            self.mask_paths.extend(mask_paths)
            self.labels.extend([idx] * len(img_paths))
        
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]  # Get label

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = mask.squeeze(0).long()
        
        return image, mask, label

# Dataset paths
base_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\segmentation_data"
train_dataset = SegmentationDataset(os.path.join(base_path, "train"), crack_types, transform_img, transform_mask)
valid_dataset = SegmentationDataset(os.path.join(base_path, "validation"), crack_types, transform_img, transform_mask)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

# Load UNet++ model
model = UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=len(crack_types)
)
model = model.to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    print(f"\U0001F680 Training UNet++ for {epochs} epochs...\n")

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{epochs}", unit="batch", leave=True)

        for images, masks, _ in progress_bar:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

        # Save best model (lowest loss)
        if running_loss < best_loss:
            best_loss = running_loss
            best_model_state = model.state_dict()
            print(f"\n🌟 Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    # Save best model
    model_save_dir = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\UNetPlusPlus_model"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "UNetPlusPlus_best_model.pth")
    torch.save(best_model_state, best_model_path)
    print(f"\n✅ Best model saved at {best_model_path}")

    # Save final model after all epochs
    final_model_path = os.path.join(model_save_dir, "UNetPlusPlus_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved at {final_model_path}")

# Run training
if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    print("🚀 Training Complete!")
