import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ─── PAGE SETTINGS ──────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# ─── CLASS NAMES ────────────────────────────────────────
class_names = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# ─── MODELS LOAD ────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn    = load_model("models/cnn_model.h5")
    resnet = load_model("models/resnet_model.h5")
    return cnn, resnet

# ─── UI ─────────────────────────────────────────────────
st.title("🚦 Traffic Sign Recognition")
st.markdown("### CNN vs ResNet50 — Deep Learning Project")
st.markdown("---")

model_choice = st.selectbox(
    "🤖 Select Model",
    ["CNN (Custom)", "ResNet50 (Transfer Learning)", "Compare Both"]
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "📸 Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png", "ppm"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)
    st.markdown("---")

    if st.button("🔍 Recognize!", use_container_width=True):

        with st.spinner("Loading models..."):
            cnn_model, resnet_model = load_models()

        img_array = np.array(image.convert("RGB"))

        # CNN preprocessing
        img_cnn = cv2.resize(img_array, (32, 32))
        img_cnn = img_cnn / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)

        # ResNet preprocessing
        img_resnet = cv2.resize(img_array, (64, 64))
        img_resnet = preprocess_input(img_resnet.astype(np.float32))
        img_resnet = np.expand_dims(img_resnet, axis=0)

        # ─── CNN PREDICTION ─────────────────────────────
        if model_choice in ["CNN (Custom)", "Compare Both"]:
            with st.spinner("CNN is predicting..."):
                cnn_pred  = cnn_model.predict(img_cnn, verbose=0)
                cnn_class = np.argmax(cnn_pred)
                cnn_conf  = np.max(cnn_pred) * 100
                cnn_name  = class_names[cnn_class]

            st.subheader("🔵 CNN Result:")
            st.success(f"**{cnn_name}**")
            st.metric("Confidence", f"{cnn_conf:.2f}%")

            top3_idx = np.argsort(cnn_pred[0])[::-1][:3]
            st.markdown("**Top 3 Predictions:**")
            for i, idx in enumerate(top3_idx):
                conf = cnn_pred[0][idx] * 100
                st.progress(int(conf), text=f"{i+1}. {class_names[idx]} — {conf:.1f}%")

            st.markdown("---")

        # ─── RESNET PREDICTION ──────────────────────────
        if model_choice in ["ResNet50 (Transfer Learning)", "Compare Both"]:
            with st.spinner("ResNet50 is predicting..."):
                resnet_pred  = resnet_model.predict(img_resnet, verbose=0)
                resnet_class = np.argmax(resnet_pred)
                resnet_conf  = np.max(resnet_pred) * 100
                resnet_name  = class_names[resnet_class]

            st.subheader("🟠 ResNet50 Result:")
            st.success(f"**{resnet_name}**")
            st.metric("Confidence", f"{resnet_conf:.2f}%")

            top3_idx = np.argsort(resnet_pred[0])[::-1][:3]
            st.markdown("**Top 3 Predictions:**")
            for i, idx in enumerate(top3_idx):
                conf = resnet_pred[0][idx] * 100
                st.progress(int(conf), text=f"{i+1}. {class_names[idx]} — {conf:.1f}%")

            st.markdown("---")

        # ─── COMPARISON ─────────────────────────────────
        if model_choice == "Compare Both":
            st.subheader("⚖️ Model Comparison:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔵 CNN Confidence", f"{cnn_conf:.2f}%")
                st.write(f"**{cnn_name}**")
            with col2:
                st.metric("🟠 ResNet50 Confidence", f"{resnet_conf:.2f}%")
                st.write(f"**{resnet_name}**")

            if resnet_conf > cnn_conf:
                st.info("🏆 ResNet50 is more confident!")
            else:
                st.info("🏆 CNN is more confident!")

# ─── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.markdown("**CNN (Custom):**")
    st.markdown("- Accuracy: 74.92%")
    st.markdown("- F1-Score: 67.62%")
    st.markdown("---")
    st.markdown("**ResNet50:**")
    st.markdown("- Accuracy: 78.37%")
    st.markdown("- F1-Score: 78.31%")
    st.markdown("---")
    st.markdown("**Dataset:** GTSRB")
    st.markdown("**Classes:** 43 Traffic Signs")
    st.markdown("**Train Images:** 39,209")
    st.markdown("**Test Images:** 12,630")