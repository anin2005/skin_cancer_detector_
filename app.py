# app.py
# ------------------------------------------------------------
# AI-Powered Skin Cancer Detector (DEMO)
# - Works with TensorFlow 2.9.x + protobuf 3.20.x (CPU-friendly)
# - Builds a tiny CNN the first time; saves to /models
# - Streamlit UI with progress bar + doctor's note + disclaimer
# ------------------------------------------------------------

import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -----------------------------pi
# Config
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "models/skin_cancer_detector_v2.keras"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -----------------------------
# Streamlit Page Setup & Styles
# -----------------------------
st.set_page_config(page_title="AI Skin Cancer Detector (Demo)", page_icon="🩺", layout="centered")

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
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">AI-Powered Skin Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a lesion image to assess risk (demo model — not for medical use)</div>', unsafe_allow_html=True)

# -----------------------------
# Helpful runtime info (shown at top for debugging)
# -----------------------------
st.caption("Environment")
st.write({
    "tensorflow": tf.__version__,
    "keras": tf.keras.__version__,
    "numpy": np.__version__,
    "model_path": MODEL_PATH
})

# Soft warning hints users toward the known-good combo
try:
    import google.protobuf as _p
    if not (_p.__version__.startswith("3.20")):
        st.warning(
            f"protobuf { _p.__version__ } detected. If you see import errors, pin protobuf==3.20.3.",
            icon="⚠️"
        )
except Exception:
    pass

# -----------------------------
# Model builder / loader
# -----------------------------
def build_image_model():
    """A tiny CNN that accepts 224x224x3 images (for DEMO ONLY)."""
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy')
    return m

@st.cache_resource(show_spinner=False)
def load_or_init_model() -> tf.keras.Model:
    """
    Try to load a saved model. If that fails OR input shape mismatches,
    build a fresh one and save it.
    """
    # Try load
    try:
        if os.path.exists(MODEL_PATH):
            m = tf.keras.models.load_model(MODEL_PATH, compile=False)
            # Basic shape sanity check
            ishape = getattr(m, "input_shape", None)
            if ishape and len(ishape) == 4 and tuple(ishape[1:4]) == (IMG_SIZE, IMG_SIZE, 3):
                # Compile if loaded without compile
                try:
                    m.compile(optimizer='adam', loss='binary_crossentropy')
                except Exception:
                    pass
                return m
    except Exception as e:
        # If load fails, we fall back to building a new model
        st.info(f"Rebuilding model due to load error: {e}")

    # Build fresh and save
    m = build_image_model()
    try:
        m.save(MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not save model to {MODEL_PATH}: {e}")
    return m

# Safely create the model; surface any exceptions in the UI
try:
    model = load_or_init_model()
except Exception as e:
    st.exception(e)
    st.stop()

st.caption(f"Model input shape: {model.input_shape}")

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_file) -> np.ndarray:
    """
    Load image -> resize to 224x224 -> normalize to [0,1] -> add batch dim -> (1,224,224,3)
    """
    image = Image.open(image_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=320)

    if st.button("🩺 Diagnose"):
        with st.spinner("Analyzing image for suspicious traits..."):
            # Simulate small delay for UX
            time.sleep(0.8)

            # Preprocess
            try:
                image = preprocess_image(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read/prepare the image: {e}")
                st.stop()

            # Predict (note: untrained model => random-ish output)
            try:
                pred = float(model.predict(image, verbose=0)[0][0])
            except Exception as e:
                st.exception(e)
                st.stop()

            # Interpret
            result = "Malignant (High Risk)" if pred > 0.5 else "Benign (Low Risk)"
            confidence = pred if pred > 0.5 else (1.0 - pred)
            confidence_percent = int(round(confidence * 100))

        # -----------------------------
        # Result Card
        # -----------------------------
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### 🧬 Diagnosis Result: **{result}**")
        st.markdown(f"**Confidence:** {confidence_percent}%")

        # Confidence Progress Bar
        st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%;"><span>{confidence_percent}%</span></div>
            </div>
        """, unsafe_allow_html=True)

        # Doctor's Note
        if result.startswith("Malignant"):
            st.error("""
**Doctor's Note (Demo):**
- The lesion is flagged as potentially malignant.
- Please consult a dermatologist for clinical evaluation.
- Additional tests (e.g., dermoscopy/biopsy) may be necessary.
""")
        else:
            st.success("""
**Doctor's Note (Demo):**
- The lesion appears likely benign.
- Keep monitoring for changes in size, color, border, or symptoms.
- Consult a doctor if you notice any changes or have concerns.
""")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("""
---
⚠️ **Important Disclaimer**

This app is a **demo**. The model here is **not trained on medical data** and is **not a diagnostic tool**.
For any health concerns, consult a licensed dermatologist and use clinically validated tools.
""")
