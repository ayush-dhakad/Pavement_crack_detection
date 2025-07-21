import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths for datasets
base_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\Image_classification"
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "validation")
test_path = os.path.join(base_path, "test")

# Load datasets
train_dataset = ImageFolder(root=train_path, transform=transform)
valid_dataset = ImageFolder(root=valid_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
num_classes = len(train_dataset.classes)

# Define the same classifier as in training
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Load trained weights
model_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\ResNet18_model\\ResNet18_trained_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

# Evaluation function
def evaluate(model, loader, name="Dataset"):
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1) * 100
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1) * 100
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1) * 100
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n📊 Evaluation Results on {name}:")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print(f"📌 Precision: {precision:.2f}%")
    print(f"📌 Recall: {recall:.2f}%")
    print(f"📌 F1 Score: {f1:.2f}%")
    print(f"📎 MCC: {mcc:.4f}")

if __name__ == "__main__":
    evaluate(model, train_loader, "Train Set")
    evaluate(model, valid_loader, "Validation Set")
    evaluate(model, test_loader, "Test Set")
