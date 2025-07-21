import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
base_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\Image_classification"
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "validation")

# Datasets
train_dataset = ImageFolder(root=train_path, transform=transform)
valid_dataset = ImageFolder(root=valid_path, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(train_dataset.classes))
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Validation function (internal)
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=20):
    print(f"🚀 Training MobileNetV2 for {epochs} epochs...\n")
    best_acc, best_epoch = 0.0, -1
    model_save_dir = os.path.join(base_path, "MobilenetV2_model")
    os.makedirs(model_save_dir, exist_ok=True)

    # Paths for saving best and final models
    best_model_path = os.path.join(model_save_dir, "MobilenetV2_best_model.pth")
    final_model_path = os.path.join(model_save_dir, "MobilenetV2_final_model.pth")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)

        # Validation after each epoch
        val_acc = evaluate_model(model, valid_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Saved best model at epoch {best_epoch} with validation accuracy: {best_acc:.2f}%")

    # Save final model after all epochs
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved after {epochs} epochs.")

    print(f"\n✅ Best model saved at epoch {best_epoch} with accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=20)
