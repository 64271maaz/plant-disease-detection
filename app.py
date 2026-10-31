import json
import numpy as np
import matplotlib
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

CONF_THRESHOLD = 0.70
IMG_SIZE = (224, 224)

TREATMENTS = {
    "healthy": "The plant looks healthy. Keep regular irrigation, balanced fertilizer, and monitor the leaves.",
    "yellowleaf": "Yellow Leaf Curl is a virus spread by whiteflies. Control whiteflies with yellow sticky traps or neem spray, and remove infected plants.",
    "mosaic": "Mosaic virus has no direct cure. Remove infected plants, sanitize tools, and use resistant varieties.",
    "spider": "For spider mites, use neem oil or a miticide spray. A strong water spray on the leaves also helps.",
    "target": "For target spot, use a copper or mancozeb fungicide. Keep good airflow between plants.",
    "septoria": "Remove older lower leaves. Use chlorothalonil or mancozeb fungicide, and reduce overhead irrigation.",
    "bacterial": "Bacterial spot has no strong chemical cure. Use copper spray, remove infected leaves, and avoid wetting the foliage.",
    "early_blight": "For early blight, use a mancozeb or chlorothalonil fungicide. Remove old leaves and use mulching.",
    "late_blight": "Late blight spreads fast. Use a fast-acting fungicide (mancozeb or metalaxyl mix) and remove infected plants quickly.",
    "mold": "For leaf mold, improve airflow, keep leaves dry, and spray copper or chlorothalonil.",
}
DEFAULT_TREATMENT = "No specific advice is available for this disease. Please consult a local agriculture expert."


def get_treatment(raw):
    r = raw.lower()
    for key, text in TREATMENTS.items():
        if key in r:
            return text
    return DEFAULT_TREATMENT


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras", compile=False)


@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()


def pretty_name(raw):
    crop, _, disease = raw.partition("___")
    if not disease:
        crop, _, disease = raw.partition("__")
    return f"{crop.replace('_', ' ')}: {disease.replace('_', ' ')}"


def looks_like_leaf(img):
    arr = np.array(img.resize(IMG_SIZE)).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green = (g > r) & (g > b) & (g > 60)
    return green.mean() > 0.15


def make_gradcam(img_arr, class_idx):
    base = model.get_layer("mobilenetv2_1.00_224")
    feature_model = tf.keras.Model(base.input, base.output)

    x = model.get_layer("rescaling")(img_arr)
    conv_out = feature_model(x, training=False)

    with tf.GradientTape() as tape:
        tape.watch(conv_out)  # base model frozen hai, isliye conv_out ko manually watch karna zaroori hai
        pooled = model.get_layer("global_average_pooling2d")(conv_out)
        logits = model.get_layer("dense")(pooled)
        score = logits[:, class_idx]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_heatmap(original, cam, alpha=0.4):
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize(original.size, Image.BILINEAR)
    heat = matplotlib.colormaps["jet"](np.array(cam_img) / 255.0)[..., :3]
    heat_img = Image.fromarray(np.uint8(heat * 255))
    return Image.blend(original, heat_img, alpha)


st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf photo, and the model will predict the disease.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf")

    if not looks_like_leaf(img):
        st.warning("This image does not look like a leaf. Please upload a clear leaf photo.")
        st.stop()

    arr = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]
    best = top3[0]
    conf = preds[best]

    if conf < CONF_THRESHOLD:
        st.warning(
            f"The model is not confident (top prediction: {pretty_name(class_names[best])}, "
            f"{conf * 100:.2f}%). Please try a clear, close-up leaf photo."
        )
    else:
        st.success(f"**Prediction:** {pretty_name(class_names[best])} ({conf * 100:.2f}%)")

        st.subheader("Treatment suggestion")
        st.info(get_treatment(class_names[best]))
        st.caption("This is general guidance, not a replacement for advice from a local agriculture expert.")

        st.subheader("Where the model is looking (Grad-CAM)")
        cam = make_gradcam(arr, best)
        st.image(
            overlay_heatmap(img, cam),
            caption="Red areas show where the model focused the most",
        )

    st.subheader("Top 3 predictions")
    for i in top3:
        st.write(f"{pretty_name(class_names[i])} ({preds[i] * 100:.2f}%)")
        st.progress(float(preds[i]))
