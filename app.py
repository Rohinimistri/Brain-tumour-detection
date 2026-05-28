import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("brain_tumor_model.h5")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Title
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to detect tumor type")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128,128))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.subheader("🔍 Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Simple interpretation
    if predicted_class == "notumor":
        st.success("✅ No Tumor Detected")
    else:
        st.warning(f"⚠️ Detected: {predicted_class}")