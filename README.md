# 🚦 Traffic Sign Recognition using Deep Learning

A machine learning project that recognizes German traffic signs using two deep learning models: Custom CNN and ResNet50.

---

## 📌 Project Overview

This project uses the GTSRB (German Traffic Sign Recognition Benchmark) dataset to train and compare two deep learning models for automatic traffic sign classification into 43 categories.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | GTSRB - German Traffic Sign Recognition Benchmark |
| **Total Images** | 51,839 |
| **Training Images** | 39,209 |
| **Test Images** | 12,630 |
| **Classes** | 43 Traffic Sign Categories |
| **Image Size** | 32×32 pixels (CNN), 64×64 pixels (ResNet50) |

---

## 🤖 Models

### Model 1 — Custom CNN
- Built from scratch using TensorFlow/Keras
- Multiple Conv2D layers with BatchNormalization
- MaxPooling and Dropout for regularization

### Model 2 — ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned on GTSRB dataset
- Last 50 layers unfrozen for fine-tuning

---

## 📈 Results

| Metric | CNN | ResNet50 |
|---|---|---|
| **Accuracy** | 74.92% | 78.37% |
| **Precision** | 63.39% | 79.01% |
| **Recall** | 74.92% | 78.37% |
| **F1-Score** | 67.62% | 78.31% |

> 🏆 ResNet50 outperforms CNN on all metrics!

---

## 🛠️ Tools & Technologies

- Python 3.10
- TensorFlow 2.15 / Keras
- Streamlit (Web App)
- OpenCV
- Scikit-learn
- Matplotlib / Seaborn

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Train CNN model:**
```bash
python cnn_model.py
```

**3. Train ResNet50 model:**
```bash
python resnet_model.py
```

**4. Compare results:**
```bash
python compare_results.py
```

**5. Run Web App:**
```bash
streamlit run app.py
```

---

## 👥 Team Members

| Name -----| Student ID |
|-----------|------------|
| mirza_saad|230180650019| |

---

