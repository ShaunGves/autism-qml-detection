"""
Autism Detection using Quantum Machine Learning (QML)
======================================================
Architecture:
  Grayscale Face Image (28x28)
        ↓
  Quantum Convolution Layer (quanvolution)
        ↓
  Quantum Feature Maps (14x14x4)
        ↓
  Flatten → Dense(64, ReLU) → Dense(2, Softmax)
        ↓
  Prediction: Autistic / Non-Autistic
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane.templates import RandomLayers

import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
n_epochs   = 30
n_layers   = 1
n_train    = 50
n_test     = 30
SAVE_PATH  = "quanvolution/"
DATASET    = "dataset/consolidated"  # update if your path differs

np.random.seed(0)
tf.random.set_seed(0)

os.makedirs(SAVE_PATH, exist_ok=True)

# ──────────────────────────────────────────────
# STEP 1: LOAD IMAGES
# ──────────────────────────────────────────────
print("Loading images...")
image_paths, labels = [], []

for label, subfolder in enumerate(["Autistic", "Non_Autistic"]):
    folder = os.path.join(DATASET, subfolder)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(folder, fname))
            labels.append(label)

labels = np.array(labels)
print(f"  Total images found: {len(image_paths)}")


def load_images(paths, width=28, height=28):
    """Load, resize, grayscale, normalize images."""
    images = []
    for p in paths:
        img = Image.open(p).resize((width, height))
        if img.mode != 'L':
            img = img.convert('L')
        arr = np.array(img) / 255.0
        images.append(arr[..., np.newaxis])   # shape: (28, 28, 1)
    return np.array(images)


images = load_images(image_paths)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
print(f"  Train: {len(train_images)} | Test: {len(test_images)}")


# ──────────────────────────────────────────────
# STEP 2: QUANTUM CIRCUIT
# ──────────────────────────────────────────────
dev = qml.device("default.qubit", wires=4)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))


@qml.qnode(dev, interface="autograd")
def circuit(phi):
    # Data encoding: rotate each qubit based on input pixel values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random trainable quantum layer (creates entanglement)
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement: returns 4 expectation values between -1 and +1
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]


# ──────────────────────────────────────────────
# STEP 3: QUANTUM CONVOLUTION (QUANVOLUTION)
# ──────────────────────────────────────────────
def quanv(image):
    """
    Slides a 2x2 quantum filter over a 28x28 image.
    Input:  (28, 28, 1)
    Output: (14, 14, 4)  — 4 quantum feature maps
    """
    out = np.zeros((14, 14, 4))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit([
                image[j,   k,   0],
                image[j,   k+1, 0],
                image[j+1, k,   0],
                image[j+1, k+1, 0],
            ])
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


# ──────────────────────────────────────────────
# STEP 4: QUANTUM PRE-PROCESSING
# ──────────────────────────────────────────────
q_train_path = SAVE_PATH + "q_train_images.npy"
q_test_path  = SAVE_PATH + "q_test_images.npy"

if os.path.exists(q_train_path) and os.path.exists(q_test_path):
    print("\nLoading saved quantum-processed images...")
    q_train_images = np.load(q_train_path)
    q_test_images  = np.load(q_test_path)
else:
    print(f"\nQuantum pre-processing {n_train} train images (this may take a while)...")
    q_train_images = []
    for idx, img in enumerate(train_images[:n_train]):
        print(f"  {idx+1}/{n_train}", end="\r")
        q_train_images.append(quanv(img))
    q_train_images = np.array(q_train_images)

    print(f"\nQuantum pre-processing {n_test} test images...")
    q_test_images = []
    for idx, img in enumerate(test_images[:n_test]):
        print(f"  {idx+1}/{n_test}", end="\r")
        q_test_images.append(quanv(img))
    q_test_images = np.array(q_test_images)

    np.save(q_train_path, q_train_images)
    np.save(q_test_path,  q_test_images)
    print("\nSaved quantum-processed images.")


# ──────────────────────────────────────────────
# STEP 5: VISUALIZE QUANTUM FEATURES
# ──────────────────────────────────────────────
def visualize_quanvolution(n_samples=4):
    n_channels = 4
    fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
    for k in range(n_samples):
        axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")
        axes[0, 0].set_ylabel("Input")
        if k != 0:
            axes[0, k].yaxis.set_visible(False)
        for c in range(n_channels):
            axes[c+1, k].imshow(q_train_images[k, :, :, c], cmap="gray")
            axes[c+1, 0].set_ylabel(f"Output [ch. {c}]")
            if k != 0:
                axes[c+1, k].yaxis.set_visible(False)
    plt.suptitle("Quantum Feature Maps vs Original Input", fontsize=13)
    plt.tight_layout()
    plt.savefig("quanvolution/feature_maps.png", dpi=150)
    plt.show()
    print("Feature map saved to quanvolution/feature_maps.png")

visualize_quanvolution()


# ──────────────────────────────────────────────
# STEP 6: CLASSICAL NEURAL NETWORK HEAD
# ──────────────────────────────────────────────
def MyModel():
    """
    Flatten → Dense(64, ReLU) → Dense(2, Softmax)
    Takes quantum feature maps (14x14x4) as input.
    Outputs probabilities for 2 classes.
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),   # 2 classes
    ])
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# STEP 7: TRAIN QUANTUM-ENHANCED MODEL
# ──────────────────────────────────────────────
print("\nTraining Quantum-Enhanced Model...")
q_model = MyModel()
q_history = q_model.fit(
    q_train_images,
    train_labels[:n_train],
    validation_data=(q_test_images, test_labels[:n_test]),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

# ──────────────────────────────────────────────
# STEP 8: TRAIN CLASSICAL BASELINE MODEL
# ──────────────────────────────────────────────
print("\nTraining Classical Baseline Model...")
c_model = MyModel()
c_history = c_model.fit(
    train_images[:n_train],
    train_labels[:n_train],
    validation_data=(test_images[:n_test], test_labels[:n_test]),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

# ──────────────────────────────────────────────
# STEP 9: COMPARE RESULTS
# ──────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(q_history.history["val_accuracy"], label="Quantum Model")
plt.plot(c_history.history["val_accuracy"], label="Classical Model")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(q_history.history["val_loss"], label="Quantum Model")
plt.plot(c_history.history["val_loss"], label="Classical Model")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("quanvolution/training_comparison.png", dpi=150)
plt.show()
print("Training comparison plot saved.")

q_val_acc = max(q_history.history["val_accuracy"])
c_val_acc = max(c_history.history["val_accuracy"])
print(f"\nBest Quantum Model Val Accuracy : {q_val_acc:.4f}")
print(f"Best Classical Model Val Accuracy: {c_val_acc:.4f}")
