# 🧠 Autism Detection using Quantum Machine Learning

A hybrid **Quantum-Classical** model that detects autism from facial images using a **Quavolution** layer (PennyLane) followed by a classical dense neural network (TensorFlow/Keras).

---

## 🔬 How It Works

```
Face Image (28×28 grayscale)
        ↓
Quantum Convolution Layer
  └─ Slides 2×2 patches over image
  └─ Each patch → 4-qubit quantum circuit
  └─ Outputs 4 expectation values per patch
        ↓
Quantum Feature Maps (14×14×4)
        ↓
Flatten → Dense(64, ReLU) → Dense(2, Softmax)
        ↓
Prediction: Autistic / Non-Autistic
```

### What makes this quantum?
- A standard CNN uses a **classical filter** to scan the image
- This model uses a **quantum circuit** instead — each 2×2 patch is encoded into 4 qubits, processed with random quantum gates and entanglement (CNOT), then measured
- The quantum layer acts as a **feature extractor**, and the classical dense layers make the final classification decision

---

## 📁 Project Structure

```
autism-qml-detection/
├── autism_detection.py      # Main script
├── requirements.txt         # Python dependencies
├── README.md
├── .gitignore
└── quanvolution/            # Auto-created: saved .npy files + plots
```

---

## 📦 Dataset

Download from Kaggle: [Autism Image Dataset](https://www.kaggle.com/datasets/cihan063/autism-image-data)

After downloading, organize it like this:-
```
dataset/
└── consolidated/
    ├── Autistic/        ← 1470 images
    └── Non_Autistic/    ← 1470 images
```

The `dataset/` folder is excluded from Git (too large). Always download it separately.

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/autism-qml-detection.git
cd autism-qml-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your dataset (see above)

# 5. Run
python autism_detection.py
```

> ⚠️ **Note:** Quantum preprocessing is slow on CPU. For 50 training images, expect **10–30 minutes**. The script saves results to `quanvolution/` so you only process once — subsequent runs load from disk automatically.

---

## 📊 Configuration

Edit the top of `autism_detection.py` to change these:

| Variable | Default | Meaning |
|----------|---------|---------|
| `n_train` | 50 | Number of training samples to quantum-process |
| `n_test` | 30 | Number of test samples to quantum-process |
| `n_epochs` | 30 | Training epochs |
| `n_layers` | 1 | Quantum circuit layers |

---

## 🧪 Tech Stack

| Library | Role |
|---------|------|
| PennyLane | Quantum circuit simulation |
| TensorFlow / Keras | Classical neural network head |
| NumPy | Numerical operations |
| Pillow | Image loading & preprocessing |
| scikit-learn | Train/test split |
| Matplotlib | Visualizations |

---

## 📈 Result

After training, two plots are saved in `quanvolution/`:
- `feature_maps.png` — quantum feature maps vs original input
- `training_comparison.png` — quantum model vs classical baseline (accuracy & loss)

---

## 💡 Key Concepts

| QML Term | Classical Equivalent |
|----------|---------------------|
| Qubit rotation (RY gate) | Weighted input |
| Random quantum layer | Hidden layer |
| CNOT gate | Feature interaction |
| Measurement (PauliZ) | Output activation |
| Expectation value | Neuron output |

---

## 🔗 References

- [PennyLane QML Tutorials](https://pennylane.ai/qml/)
- [Quanvolutional Neural Networks (Henderson et al., 2020)](https://arxiv.org/abs/1904.04767)
