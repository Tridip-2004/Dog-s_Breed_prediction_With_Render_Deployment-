import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:\\Users\\tridi\\OneDrive\\Desktop\\render-deployment\\dog_breed_classifier.h5")

model = load_model()

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- IMAGE PREPROCESS ----------------
IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- UI ----------------
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image to predict its breed.")

uploaded_file = st.file_uploader(
    "Choose a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Breed"):
        with st.spinner("Classifying..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            confidence = np.max(prediction) * 100
            predicted_class = class_names[np.argmax(prediction)]

        st.success(f"🐕 **Breed:** {predicted_class}")
        st.info(f"🔍 **Confidence:** {confidence:.2f}%")
