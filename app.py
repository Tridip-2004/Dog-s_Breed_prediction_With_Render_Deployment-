import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dog_breed_classifier.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open(CLASS_PATH, "r") as f:
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
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Predict Breed"):
        with st.spinner("Classifying..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)

            confidence = float(np.max(prediction) * 100)
            predicted_class = class_names[int(np.argmax(prediction))]

        st.success(f"🐕 **Breed:** {predicted_class}")
        st.info(f"🔍 **Confidence:** {confidence:.2f}%")

