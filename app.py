import os
import time
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import streamlit as st

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "models/skin_cancer_detector_v2.keras"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -----------------------------
# Model Loader
# -----------------------------
def build_image_model():
    """A simple CNN that accepts 224x224x3 images (demo only)."""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

def get_model():
    """Load model if valid, else build and save a new one."""
    try:
        if os.path.exists(MODEL_PATH):
            m = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if len(m.input_shape) == 4 and m.input_shape[1:4] == (IMG_SIZE, IMG_SIZE, 3):
                return m
    except Exception:
        pass
    m = build_image_model()
    m.save(MODEL_PATH)
    return m

model = get_model()

# -----------------------------
# Styles
# -----------------------------
st.set_page_config(page_title="AI Skin Cancer Detector", page_icon="🩺", layout="centered")

st.markdown("""
<style>
body, .main { background-color: #e6f0ff; }
.title { text-align:center; font-size: 42px; color:#003366; font-weight:800; margin-bottom:4px; }
.subtitle { text-align:center; font-size: 18px; color:#336699; margin-bottom: 12px; }
.result-card {
  background-color: #fff; border-radius: 16px; padding: 20px; margin-top: 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
.confidence-bar { height: 20px; background-color:#cce0ff; border-radius: 12px; overflow:hidden; margin-top:10px; }
.confidence-fill { height:100%; background: linear-gradient(90deg, #0033cc, #3399ff);
  text-align:right; padding-right:10px; color:white; line-height:20px; font-weight:700; transition: width 0.8s ease; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">AI-Powered Skin Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a lesion image to assess risk (demo model)</div>', unsafe_allow_html=True)
st.caption(f"Model input shape: {model.input_shape}")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

def preprocess_image(image_file):
    """Load image -> resize -> normalize -> (1,224,224,3)."""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)

    if st.button("🩺 Diagnose"):
        with st.spinner("Analyzing Image for Cancerous Traits..."):
            time.sleep(1)

            # Preprocess
            image = preprocess_image(uploaded_file)

            # Predict
            prediction = model.predict(image, verbose=0)[0][0]

            # Interpret
            result = "Malignant (High Risk)" if prediction > 0.5 else "Benign (Low Risk)"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            confidence_percent = int(confidence * 100)

        # -----------------------------
        # Result Card
        # -----------------------------
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### 🧬 Diagnosis Result: **{result}**")
        st.markdown(f"**Cancer Probability:** {confidence_percent}%")

        # Confidence Progress Bar
        st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%;">{confidence_percent}%</div>
            </div>
        """, unsafe_allow_html=True)

        # Doctor's Note
        if result.startswith("Malignant"):
            st.error("""
                **Doctor's Note (Demo):**
                - The lesion is flagged as potentially malignant.
                - Please consult a dermatologist for clinical evaluation.
                - Further tests (like biopsy) may be required.
            """)
        else:
            st.success("""
                **Doctor's Note (Demo):**
                - The lesion appears likely benign.
                - Regular monitoring is advised.
                - Consult a doctor if you notice changes.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("""
---
⚠️ **Important Disclaimer:**  
This demo is **not a medical tool**. The model here is randomly initialized and not trained on medical data.  
For real diagnosis, use a clinically validated AI model and consult a licensed dermatologist.
""")
