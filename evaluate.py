import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
IMAGE_SIZE = (200, 200)
NUM_CLASSES = 6
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
VAL_DIR = r"C:\Users\K B Taredars\Downloads\archive\NEU-DET\validation"

# Import Model Architecture
from defect_detection_phase1 import DefectClassifier

# Preprocessing Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model using device: {device}")

    # Load Model
    model = DefectClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("defect_detector_model.pth", map_location=device))
    model.eval()

    all_images = glob.glob(os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True)
    
    y_true = []
    y_pred = []

    print(f"Processing {len(all_images)} validation images...")

    with torch.no_grad():
        for img_path in all_images:
            folder_name = os.path.basename(os.path.dirname(img_path))
            if folder_name not in CLASS_NAMES:
                continue

            true_label = CLASS_NAMES.index(folder_name)
            y_true.append(true_label)

            # Image Prediction
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            y_pred.append(predicted.item())

    # 1. Print Text Classification Report
    print("\n=======================================================")
    print("              CLASSIFICATION REPORT                    ")
    print("=======================================================\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("NEU-DET Defect Classifier - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()