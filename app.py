import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_model.h5")

# Load the model with caching for faster inference
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# Load class labels
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load, resize, normalize, and prepare image for model prediction"""
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image_class(image):
    """Predict plant disease class from the uploaded image"""
    processed_img = load_and_preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

# Streamlit App UI
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Set background color to green */
    body {
        background-color: #E8F5E9; /* Light green background */
    }
    .stApp {
        background-color: #E8F5E9;
    }
    /* Style the Classify button */
    div.stButton > button:first-child {
        background-color: #4CAF50; /* Green color */
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%; /* Full width */
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    /* Style the file uploader button */
    .stFileUploader > div > div > div > button {
        background-color: #4CAF50; /* Green color */
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stFileUploader > div > div > div > button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    /* Style the success message */
    .stSuccess {
        font-size: 18px;
        color: #4CAF50; /* Green color */
        font-weight: bold;
    }
    /* Style the sidebar */
    .css-1d391kg {
        background-color: #C8E6C9; /* Light green sidebar */
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection")
st.sidebar.write("Upload a plant leaf image to classify the disease.")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("- Upload an image 📷\n- Click 'Classify' 🔍\n- Get the result instantly! ✅")

# Main content
st.title("🌱 Plant Disease Classifier")
st.write("Upload an image of a plant leaf to detect potential diseases.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Classification Result")
        if st.button("Classify", key="classify_button", use_container_width=True):
            with st.spinner("Classifying..."):
                prediction = predict_image_class(uploaded_image)
                st.markdown(f"<p class='stSuccess'>🟢 Prediction: {prediction}</p>", unsafe_allow_html=True)