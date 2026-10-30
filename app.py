import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

def pretty_name(raw):
    # "Tomato___Late_blight" -> "Tomato: Late blight"
    crop, _, disease = raw.partition("___")
    if not disease:
        crop, _, disease = raw.partition("__")
    return f"{crop.replace('_', ' ')}: {disease.replace('_', ' ')}"

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf photo, and the model will predict the disease.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf")

    arr = np.array(img.resize((224, 224)), dtype="float32")
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]

    st.success(f"**Prediction:** {pretty_name(class_names[top3[0]])} ({preds[top3[0]] * 100:.2f}%)")

    st.subheader("Top 3 predictions")
    for i in top3:
        st.write(f"{pretty_name(class_names[i])}: {preds[i] * 100:.2f}%")
        st.progress(float(preds[i]))
