import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Page Configuration
st.set_page_config(
    page_title="Steel Defect AI Inspector",
    page_icon="🔍",
    layout="centered"
)

# Configuration & Constants
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

# Import Architecture
from defect_detection_phase1 import DefectClassifier

# Preprocessing Pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("defect_detector_model.pth", map_location=device))
    model.eval()
    return model, device

# App UI Header
st.title("🏭 Automated Steel Defect Detection")
st.markdown("Upload a steel surface image to inspect and classify manufacturing defects in real-time.")

# Sidebar Information
st.sidebar.header("Supported Defects")
for cls in CLASS_NAMES:
    st.sidebar.markdown(f"- **{cls.capitalize()}**")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a steel surface image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Surface Image", use_container_width=True)

    # Load Model & Predict
    model, device = load_model()
    input_tensor = transform_pipeline(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    # Display Prediction Results
    with col2:
        st.subheader("Inspection Result")
        st.metric(
            label="Detected Defect", 
            value=predicted_class.upper(), 
            delta=f"{confidence_score:.2f}% Confidence"
        )

        st.markdown("---")
        st.write("### Probability Distribution")
        for idx, class_name in enumerate(CLASS_NAMES):
            prob = probabilities[idx].item() * 100
            st.write(f"**{class_name.capitalize()}**: {prob:.1f}%")
            st.progress(prob / 100)