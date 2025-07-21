# evaluate_mobilenetv2.py

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
base_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\Image_classification"
train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "validation")
test_path = os.path.join(base_path, "test")

# Datasets
train_dataset = ImageFolder(root=train_path, transform=transform)
valid_dataset = ImageFolder(root=valid_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load Model
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, len(train_dataset.classes))
)
model_path = "D:\\VS-Code\\Python\\Pavement_Crack_Detection\\MobilenetV2_model\\MobilenetV2_trained_model.pth"
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

    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average="weighted") * 100
    f1 = f1_score(y_true, y_pred, average="weighted") * 100
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n📊 Evaluation Results on {name}:")
    print(f"🎯 Accuracy: {acc:.2f}%")
    print(f"📌 Precision: {precision:.2f}%")
    print(f"📌 Recall: {recall:.2f}%")
    print(f"📌 F1 Score: {f1:.2f}%")
    print(f"📎 MCC: {mcc:.4f}")

if __name__ == "__main__":
    evaluate(model, train_loader, "Train Set")
    evaluate(model, valid_loader, "Validation Set")
    evaluate(model, test_loader, "Test Set")
