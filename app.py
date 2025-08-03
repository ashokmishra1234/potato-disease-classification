import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
MODEL_PATH = r"./potatoes.keras"  # Update with your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Class names - update based on your model's training
class_names = ['Early Blight','Healthy','Late Blight']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))  # Adjust size based on your model's input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🥔 Potato Leaf Disease Classification")

st.write("Upload an image of a potato leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

