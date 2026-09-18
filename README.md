# 🏭 Industrial Steel Defect Detection System

An end-to-end Deep Learning application designed to classify manufacturing surface defects on steel sheets using the **NEU-DET** dataset. Built with **PyTorch** for model training and **Streamlit** for real-time visual inspection via an interactive web application.

---

## 📌 Project Overview
Steel manufacturing plants require high-precision visual quality control. Manual inspection is slow and prone to error. This project automates defect recognition across 6 common surface defect categories with confidence scoring and probability distributions.

### 🔍 Supported Defect Classes
* **Crazing** (Baarik dararein)
* **Inclusion** (External impurities)
* **Patches** (Surface blemishes)
* **Pitted Surface** (Small surface cavities)
* **Rolled-in Scale** (Oxide scale pressed into metal)
* **Scratches** (Abrasive lines)

---

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.11
* **Deep Learning Framework:** PyTorch & Torchvision
* **Computer Vision:** OpenCV & Pillow
* **Data Processing & Evaluation:** Scikit-Learn, Seaborn, Matplotlib
* **Web Dashboard:** Streamlit
* **Version Control:** Git & GitHub

---

## 🚀 Key Features
1. **Custom CNN Architecture:** Trained for 10 epochs on normalized $200 \times 200$ images.
2. **Model Evaluation:** Includes scripts for generating Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).
3. **CLI & Visual Inspection:** Interactive command-line testing with dynamic Matplotlib image overlays.
4. **Interactive Web App:** Drag-and-drop web dashboard with real-time class probability breakdown.

---

## 📁 Repository Structure
```text
├── app.py                      # Interactive Streamlit Web Application
├── defect_detection_phase1.py  # PyTorch Model Architecture Definition
├── predict.py                  # CLI Random Sample Prediction Script
├── visual_predict.py           # GUI Matplotlib Popup Visualizer
├── evaluate.py                 # Evaluation Metrics & Confusion Matrix Generator
├── defect_detector_model.pth   # Saved PyTorch Model Weights
├── requirements.txt            # Project Dependencies for Deployment
└── README.md                   # Project Documentation