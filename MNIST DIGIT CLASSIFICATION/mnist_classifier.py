"""
MNIST Digit Classification with Neural Network
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 0. Reproducibility & output setup
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load the MNIST dataset from keras.datasets
# ---------------------------------------------------------------------------
def load_data():
    print("Step 1/6: Loading MNIST dataset from keras.datasets ...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"  Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# 2. Preprocess the data
# ---------------------------------------------------------------------------
def preprocess_data(x_train, y_train, x_test, y_test, val_split=0.1):
    print("Step 2/6: Preprocessing (normalizing, flattening, one-hot encoding) ...")

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten 28x28 images into 784-length vectors for the dense network
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))

    # One-hot encode labels (10 classes: digits 0-9)
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # Carve out a validation split from the training set
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train_cat[:val_size]
    x_train_final, y_train_final = x_train[val_size:], y_train_cat[val_size:]

    print(f"  Train: {x_train_final.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    return (x_train_final, y_train_final), (x_val, y_val), (x_test, y_test_cat, y_test)


# ---------------------------------------------------------------------------
# 3. Build the neural network
# ---------------------------------------------------------------------------
def build_model():
    print("Step 3/6: Building the neural network architecture ...")
    model = keras.Sequential(
        [
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_dense_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 4. Train the model
# ---------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128):
    print("Step 4/6: Training the model ...")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ---------------------------------------------------------------------------
# 5. Evaluate the model
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test_cat, y_test_raw):
    print("Step 5/6: Evaluating on the test set ...")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n  Classification Report:")
    print(classification_report(y_test_raw, y_pred, digits=4))

    return y_pred, y_pred_probs, test_loss, test_acc


# ---------------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------------
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves -> {path}")


def plot_confusion_matrix(y_test_raw, y_pred):
    cm = confusion_matrix(y_test_raw, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix -> {path}")


def plot_sample_predictions(x_test, y_test_raw, y_pred, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    idxs = np.random.choice(len(x_test), n, replace=False)
    for ax, idx in zip(axes.flatten(), idxs):
        img = x_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        correct = y_test_raw[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"True: {y_test_raw[idx]} | Pred: {y_pred[idx]}", color=color, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved sample predictions -> {path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    (x_train_f, y_train_f), (x_val, y_val), (x_test_p, y_test_cat, y_test_raw) = preprocess_data(
        x_train, y_train, x_test, y_test
    )

    model = build_model()

    history = train_model(model, x_train_f, y_train_f, x_val, y_val, epochs=20, batch_size=128)

    y_pred, y_pred_probs, test_loss, test_acc = evaluate_model(
        model, x_test_p, y_test_cat, y_test_raw
    )

    print("Step 6/6: Generating visualizations ...")
    plot_training_history(history)
    plot_confusion_matrix(y_test_raw, y_pred)
    plot_sample_predictions(x_test_p, y_test_raw, y_pred)

    model_path = os.path.join(OUTPUT_DIR, "mnist_model.keras")
    model.save(model_path)
    print(f"\nModel saved -> {model_path}")
    print(f"Final Test Accuracy: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")
    print("\nDone. All artifacts are in the 'outputs/' directory.")


if __name__ == "__main__":
    main()
