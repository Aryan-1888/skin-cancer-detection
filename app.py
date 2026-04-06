import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

from PIL import Image
import numpy as np
import tensorflow.lite as tflite

# =========================
# LOAD TFLITE MODEL
# =========================
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        return "Malignant", prediction
    else:
        return "Benign", 1 - prediction

# =========================
# UI
# =========================
st.title("🩺 Skin Cancer Detection System")
st.write("Upload a skin image to check whether it is Benign or Malignant.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# =========================
# LOGIC
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(image)

        st.subheader("Result:")

        if label == "Malignant":
            st.error(f"⚠️ Malignant ({confidence*100:.2f}%)")
        else:
            st.success(f"✅ Benign ({confidence*100:.2f}%)")

        st.warning("⚠️ This is an AI-based prediction and not a medical diagnosis.")