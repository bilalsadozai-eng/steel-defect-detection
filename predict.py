import glob
import os
import random  # Random image select karne ke liye
import torch
from PIL import Image
from torchvision import transforms

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

from defect_detection_phase1 import DefectClassifier

transform_pipeline = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DefectClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load("defect_detector_model.pth", map_location=device)
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform_pipeline(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("\n====================================")
    print("      DEFECT DETECTION RESULT       ")
    print("====================================")
    print(f"Image File: {os.path.basename(image_path)}")
    print(
        f"Actual Folder/Class: {os.path.basename(os.path.dirname(image_path))}"
    )
    print(f"Predicted Defect: {predicted_class.upper()}")
    print(f"Confidence Score: {confidence_score:.2f}%")
    print("====================================\n")


if __name__ == "__main__":
    VAL_DIR = r"C:\Users\K B Taredars\Downloads\archive\NEU-DET\validation"

    all_images = glob.glob(
        os.path.join(VAL_DIR, "**", "*.jpg"), recursive=True
    )

    if len(all_images) > 0:
        TEST_IMAGE_PATH = random.choice(all_images)  # Har baar alag image chunega
        predict_image(TEST_IMAGE_PATH)
    else:
        print(f"Validation folder mein koi image nahi mili: {VAL_DIR}")
        