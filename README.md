# 🕵️‍♂️ Fake vs Real Face Detection using Deep Learning

## 📌 Project Overview
This project focuses on detecting **real vs AI-generated (deepfake) faces** using a deep learning model built with **ResNet18**.  
The model achieves **95% accuracy** on the test set and is deployed as a **Streamlit web app** that supports both **single** and **batch image uploads**.  

---

## 🚀 Features
- 🔍 Classifies faces as **Real** or **Fake (AI-generated deepfake)**
- 📦 Supports **single & multiple image uploads**
- 🖼️ Provides **Grad-CAM visualizations** to interpret predictions
- ⚡ Utilizes **GPU acceleration** for fast training
- 🌐 Fully deployable Streamlit app (Streamlit Cloud, HuggingFace Spaces, or Docker)

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Web Framework:** Streamlit  
- **Computer Vision:** OpenCV  
- **Model Architecture:** ResNet18  
- **Visualization:** Grad-CAM  
- **Hardware Acceleration:** NVIDIA RTX 2060 (GPU)  

---

## 📂 Dataset
- Source: **140k+ images dataset** containing real and fake (deepfake) faces.  
- Balanced sampling for training:
  - **20,000** training images  
  - **4,000** validation images  
  - **4,000** test images  
- Applied preprocessing (resizing, normalization, augmentations).

---

## 🧠 Model Details
- Base model: **ResNet18** (pre-trained on ImageNet).  
- Final fully connected layer modified for **binary classification**.  
- Loss function: **CrossEntropyLoss**.  
- Optimizer: **Adam**.  
- Learning rate scheduling applied for optimal convergence.  

---

## ⚡ Training Setup
- Framework: **PyTorch**  
- Epochs: **25**  
- Batch size: **64**  
- Training device: **GPU (NVIDIA RTX 2060)**  
- Reduced training time by **70%** using GPU acceleration.  

---

## 📊 Evaluation
Model performance was evaluated on the **test set** using:  
- ✅ Accuracy: **95.27%**  
- ✅ Precision, Recall, F1-score  
- ✅ Confusion Matrix  
- ✅ Error analysis on misclassified samples  

---

## 🔎 Interpretability with Grad-CAM
- Integrated **Grad-CAM** to highlight important facial regions influencing predictions.  
- Helps in understanding **why** the model predicts a face as real or fake.  

---

## 🌐 Streamlit Web App
The interactive web app allows:  
- Upload of **single or multiple images**.  
- **Real-time classification** of uploaded faces.  
- **Grad-CAM heatmaps** for each prediction.  

### Run locally:
```bash
# Clone the repository
git clone https://github.com/your-username/fake-vs-real-face-detection.git
cd fake-vs-real-face-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
