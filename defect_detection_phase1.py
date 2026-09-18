# ============================================================
# Automated Defect Detection System
# OpenCV Preprocessing + PyTorch Model (NEU-DET Dataset)
# ============================================================

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ------------------------------------------------------------
# 1. Paths & Configuration Setup
# ------------------------------------------------------------
TRAIN_DIR = r"C:\Users\K B Taredars\Downloads\archive\NEU-DET\train"
VAL_DIR = r"C:\Users\K B Taredars\Downloads\archive\NEU-DET\validation"

BATCH_SIZE = 16
IMAGE_SIZE = (200, 200)
NUM_CLASSES = 6
EPOCHS = 10
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ------------------------------------------------------------
# 2. Synthetic Defect Generator & OpenCV Preprocessing Pipeline
# ------------------------------------------------------------
def create_synthetic_image(width=600, height=400):
    image = np.full((height, width), 180, dtype=np.uint8)
    noise = np.random.normal(0, 5, (height, width))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Simulated Defects
    cv2.circle(image, center=(150, 120), radius=25, color=50, thickness=-1)
    cv2.rectangle(image, pt1=(300, 80), pt2=(420, 105), color=40, thickness=-1)

    points = np.array(
        [[450, 250], [490, 230], [520, 260], [500, 300], [460, 290]]
    )
    cv2.fillPoly(image, [points], color=30)
    cv2.circle(image, center=(200, 300), radius=15, color=45, thickness=-1)

    return image


def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return gray, blurred, thresholded


def detect_defects(original_image, thresholded):
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    detected_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2
        )
        cv2.putText(
            annotated_image,
            f"Defect {detected_count + 1}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        detected_count += 1

    return annotated_image, contours, detected_count


# ------------------------------------------------------------
# 3. PyTorch Dataset & DataLoaders (NEU-DET)
# ------------------------------------------------------------
class NEUDetDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        images_path = os.path.join(root_dir, "images")
        search_path = (
            os.path.join(images_path, "*", "*.jpg")
            if os.path.exists(images_path)
            else os.path.join(root_dir, "*", "*.jpg")
        )
        found_images = glob.glob(search_path)

        for img_path in found_images:
            class_name = os.path.basename(os.path.dirname(img_path))
            if class_name in CLASS_TO_IDX:
                self.image_paths.append(img_path)
                self.labels.append(CLASS_TO_IDX[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform_pipeline = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ------------------------------------------------------------
# 4. Neural Network Architecture (PyTorch)
# ------------------------------------------------------------
class DefectClassifier(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super(DefectClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE[0] // 8) * (IMAGE_SIZE[1] // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# 5. Model Training Loop
# ------------------------------------------------------------
def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training on NEU-DET Dataset...")
    print("---------------------------------------")

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100

        # Validation Step
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = (val_correct / max(val_total, 1)) * 100

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    # Model Weights Save Karein
    torch.save(model.state_dict(), "defect_detector_model.pth")
    print("\nModel saved successfully as 'defect_detector_model.pth'!")


# ------------------------------------------------------------
# Main Program Entry Point
# ------------------------------------------------------------
def main():
    print("Starting Automated Defect Detection System...")

    # Step 1: Run OpenCV Synthetic Defect Pipeline
    original_image = create_synthetic_image()
    gray, blurred, thresholded = preprocess_image(original_image)
    annotated_image, _, detected_count = detect_defects(
        original_image, thresholded
    )
    print(f"Synthetic Defects Detected: {detected_count}")

    # Step 2: Run PyTorch Pipeline for NEU-DET
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = NEUDetDataset(
        root_dir=TRAIN_DIR, transform=transform_pipeline
    )
    val_dataset = NEUDetDataset(root_dir=VAL_DIR, transform=transform_pipeline)

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        model = DefectClassifier(num_classes=NUM_CLASSES).to(device)
        train_model(model, train_loader, val_loader, device)
    else:
        print(
            f"Dataset images not found in {TRAIN_DIR}. Check directory path."
        )


if __name__ == "__main__":
    main()