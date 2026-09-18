# 🏭 Automated Steel Defect Detection System
### *Industrial-Grade Computer Vision & Deep Learning Pipeline*

[![Live Demo](https://img.shields.io/badge/Streamlit-Live--App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://bilal-steel-defect-detection.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end deep learning framework engineered for automated surface inspection in steel manufacturing processes. Built with PyTorch and deployed on Streamlit Cloud.

</div>

---

## 📌 Executive Summary

Manual quality assurance in hot/cold rolling steel mills is prone to human error, slow throughput, and safety hazards. This system automates the inspection process by leveraging custom Convolutional Neural Networks (CNNs) to classify surface defects with high confidence in real-time.

---

## 🎯 Defect Taxonomy

The model is trained to classify **6 primary industrial steel defect types**:

| Defect Class | Physical Characteristics | Severity Impact |
| :--- | :--- | :--- |
| **Crazing** | Micro-fissures caused by rapid thermal cooling | Medium |
| **Inclusion** | Foreign non-metallic matter embedded on surface | High |
| **Patches** | Localized oxidation or uneven coating areas | Low - Medium |
| **Pitted Surface** | Small cavities or indentations due to mechanical wear | High |
| **Rolled-in Scale** | Oxide scale pressed into metal during rolling | Critical |
| **Scratches** | Linear surface abrasion from mechanical handling | Low |

---

## 🏗️ System Architecture

┌─────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│  Image Input    │ ───> │ Torchvision Transforms  │ ───> │ Custom PyTorch CNN     │
│ (JPG, PNG, JPEG)│      │ Resize, Tensor, Norm    │      │ (DefectClassifier)     │
└─────────────────┘      └─────────────────────────┘      └───────────┬────────────┘
│
┌─────────────────┐      ┌─────────────────────────┐                  │
│ Streamlit UI    │ <─── │ Softmax Probabilities   │ <──────────────────┘
│ Metrics & Charts│      │ Confidence & Metrics    │
└─────────────────┘      └─────────────────────────┘


---

## ⚡ Quickstart & Local Deployment

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bilalsadozai-eng/steel-defect-detection.git](https://github.com/bilalsadozai-eng/steel-defect-detection.git)
   cd steel-defect-detection
