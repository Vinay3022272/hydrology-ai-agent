# 🌊 Mahanadi River — CNN-LSTM U-Net Deep Learning Model

A deep learning project for hydrological prediction on the **Mahanadi River basin** using a hybrid **CNN-LSTM U-Net** architecture.

---

## 📁 Project Files

> Large model and data files are hosted on Google Drive due to GitHub's file size limits.

| File | Description | Download |
|------|-------------|----------|
| `mahanadi_cnnlstm_unet11.keras` | Trained CNN-LSTM U-Net model weights | [Download](https://drive.google.com/file/d/1uzTslfbVj9alNrrhWvcYVLZWG0FzCkhH/view?usp=drivesdk) |
| `Mahanadi_DeepLearning_Data.npz` | Compressed NumPy dataset (features + labels) | [Download](https://drive.google.com/file/d/1qYlM1WMS7geesO3nqQh9nIk2MqEwaA8H/view?usp=drivesdk) |
| `y_clean.npy` | Cleaned target/output array | [Download](https://drive.google.com/file/d/1k9Ck1hp23w9Rfh3dhOH3HDrQNuvrGojx/view?usp=drivesdk) |
| `scaler_X.pkl` | Fitted scaler for input features (X) | [Download](https://drive.google.com/file/d/1emtmoE1tDnnr761TYdS426jUJUHiIlna/view?usp=drivesdk) |
| `scaler_y.pkl` | Fitted scaler for target variable (y) | [Download](https://drive.google.com/file/d/1C2Mj0UHxpUFS6EmPbRiCJosp8Jcz9u2F/view?usp=drivesdk) |

---

## 🏗️ Model Architecture

The model combines:
- **CNN** — spatial feature extraction from river basin data
- **LSTM** — temporal/sequential pattern learning
- **U-Net** — encoder-decoder structure for dense predictions

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Required packages: `tensorflow`, `keras`, `numpy`, `scikit-learn`, `joblib`

### 3. Download Data & Model Files

Download all files from the table above and place them in the project root directory.

### 4. Load and Run the Model

```python
import numpy as np
import joblib
from tensorflow import keras

# Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Load data
data = np.load("Mahanadi_DeepLearning_Data.npz")
y_clean = np.load("y_clean.npy")

# Load trained model
model = keras.models.load_model("mahanadi_cnnlstm_unet11.keras")

# Predict
X = data['X']  # adjust key as needed
X_scaled = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
predictions = model.predict(X_scaled)
predictions_original = scaler_y.inverse_transform(predictions)
```

---

## 📊 Dataset

The dataset covers hydrological and/or meteorological variables for the **Mahanadi River basin**, India. Features are preprocessed and normalized using `scaler_X.pkl`, and the target variable is normalized using `scaler_y.pkl`.

---

## 📋 Requirements

```
tensorflow>=2.10
keras
numpy
scikit-learn
joblib
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Data sourced from the Mahanadi River basin hydrology studies
- Model built with TensorFlow / Keras
