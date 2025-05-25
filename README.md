# Cassava Leaf Disease Classification – Custom CNN

This project is a deep learning model built **from scratch** (no pretrained weights) for classifying cassava leaf diseases using the **Cassava Leaf Disease Classification** dataset. It leverages a custom Convolutional Neural Network (CNN) with TensorFlow/Keras and is optimized for balanced accuracy across five cassava disease categories.

---

## 📂 Dataset

The dataset contains high-resolution images of cassava leaves, labeled with one of five disease classes:

1. Cassava Bacterial Blight (CBB)
2. Cassava Brown Streak Disease (CBSD)
3. Cassava Green Mottle (CGM)
4. Cassava Mosaic Disease (CMD)
5. Healthy

> 📌 Images are stored in `train_images/` and labeled via `train.csv`.

---

## 🧠 Model Architecture

A custom CNN with the following layers:
- 3 convolutional blocks with:
  - `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout`
- Fully connected layers with `Dense` and `Dropout`
- Final output with softmax activation for 5-class classification

---

## 🧪 Data Augmentation

To improve generalization, strong image augmentations are applied:
- Rotation
- Zoom
- Width/height shift
- Shearing
- Horizontal & vertical flip

---

## 🚀 Training Details

- Image size: 224×224
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: Up to 30
- Early stopping and learning rate reduction used
- Train/Val/Test split with stratification

---

## 📊 Performance

After training, the model achieves high classification accuracy on the test set with proper generalization. Evaluation is done using:

- Accuracy
- Loss curves
- Optionally: confusion matrix, F1-score

---

## 🛠️ Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- Pandas, NumPy, scikit-learn
- Jupyter Notebook (recommended)

Install dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn
