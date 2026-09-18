import os
import glob
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 1. Configuration Setup
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

# Import Architecture from Phase 1
from defect_detection_phase1 import DefectClassifier

# Preprocessing Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_and_visualize(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = DefectClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("defect_detector_model.pth", map_location=device))
    model.eval()

    # Load Original Image (for plotting) & Tensor Image (for Model)
    original_pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform_pipeline(original_pil_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    actual_class = os.path.basename(os.path.dirname(image_path))

    # Terminal Printout
    print("\n====================================")
    print("      VISUAL DEFECT INSPECTION      ")
    print("====================================")
    print(f"Image File: {os.path.basename(image_path)}")
    print(f"Actual Class: {actual_class.upper()}")
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence_score:.2f}%")
    print("====================================\n")

    # Matplotlib Popup Window Setup
    plt.figure(figsize=(7, 7))
    plt.imshow(original_pil_img)
    plt.axis('off')

    # Status Color Setup (Green if match, Red/Orange if mismatch)
    is_correct = (predicted_class == actual_class)
    banner_color = "darkgreen" if is_correct else "red"

    title_text = f"PREDICTED: {predicted_class.upper()} ({confidence_score:.1f}%)\nACTUAL: {actual_class.upper()}"
    
    plt.title(
        title_text,
        fontsize=14,
        fontweight="bold",
        color=banner_color,
        pad=15
    )

    # Display Window
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    all_images = glob.glob(os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True)

    if len(all_images) > 0:
        random_image_path = random.choice(all_images)
        predict_and_visualize(random_image_path)
    else:
        print(f"Validation folder mein koi image nahi mili: {VAL_DIR}")