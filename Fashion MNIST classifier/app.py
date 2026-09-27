"""
Streamlit app for the Fashion MNIST classifier.

Run with:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from PIL import Image, ImageOps
import pandas as pd

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

MODEL_PATH = "fashion_mnist_model.keras"

st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="👕", layout="centered")


# ----------------------------------------------------------------------
# Data / model helpers (cached so they only run once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))
    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource(show_spinner=False)
def get_model(epochs: int):
    """Load a saved model if present, otherwise train one and cache it."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except (IOError, OSError, ValueError):
        pass

    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    model.save(MODEL_PATH)
    return model, {"history": history.history, "test_acc": test_acc, "test_loss": test_loss}


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image to a normalized 28x28x1 array matching Fashion MNIST format."""
    img = img.convert("L")            # grayscale
    img = ImageOps.invert(img) if is_light_background(img) else img
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    return arr


def is_light_background(img: Image.Image) -> bool:
    """Fashion MNIST images are white-on-black; auto-invert photos with a light background."""
    arr = np.array(img)
    return arr.mean() > 127


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("👕 Fashion MNIST Classifier")
st.write(
    "A CNN trained on the Fashion MNIST dataset, classifying clothing into "
    "10 categories. Upload an image or try a random test sample below."
)

with st.sidebar:
    st.header("Model")
    epochs = st.slider("Training epochs (first run only)", min_value=3, max_value=20, value=10)
    if st.button("Retrain model"):
        st.cache_resource.clear()
        st.rerun()

with st.spinner("Loading model — training on first run may take a minute..."):
    model, train_info = get_model(epochs)

if train_info:
    st.success(f"Model trained — test accuracy: {train_info['test_acc']:.2%}")
    with st.expander("Training curves"):
        hist_df = pd.DataFrame({
            "accuracy": train_info["history"]["accuracy"],
            "val_accuracy": train_info["history"]["val_accuracy"],
        })
        st.line_chart(hist_df)
        loss_df = pd.DataFrame({
            "loss": train_info["history"]["loss"],
            "val_loss": train_info["history"]["val_loss"],
        })
        st.line_chart(loss_df)

tab1, tab2 = st.tabs(["📤 Upload an image", "🎲 Random test sample"])

with tab1:
    uploaded = st.file_uploader("Upload a clothing image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded image", use_container_width=True)

        arr = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))

        with col2:
            st.image(arr.reshape(28, 28), caption="Model input (28x28)", use_container_width=True, clamp=True)

        st.subheader(f"Prediction: **{CLASS_NAMES[top_idx]}** ({preds[top_idx]:.1%} confidence)")
        st.bar_chart(pd.DataFrame({"confidence": preds}, index=CLASS_NAMES))

with tab2:
    (_, _), (test_images, test_labels) = load_data()
    if st.button("Pick a random test image"):
        idx = np.random.randint(0, len(test_images))
        st.session_state["sample_idx"] = idx

    idx = st.session_state.get("sample_idx", 0)
    img_arr = test_images[idx]
    true_label = test_labels[idx]

    preds = model.predict(img_arr.reshape(1, 28, 28, 1), verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_arr.reshape(28, 28), caption=f"True label: {CLASS_NAMES[true_label]}", use_container_width=True, clamp=True)
    with col2:
        color = "green" if pred_idx == true_label else "red"
        st.markdown(f"Prediction: :{color}[**{CLASS_NAMES[pred_idx]}**] ({preds[pred_idx]:.1%})")
        st.bar_chart(pd.DataFrame({"confidence": preds}, index=CLASS_NAMES))

st.caption("Model: 3x Conv2D + MaxPooling, Dense(64), Dropout(0.3), Dense(10, softmax)")
