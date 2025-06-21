# SkinDiseaseClassifier-MedicineRecommendor
Here's a polished **README.md** for your project based on the notebook you uploaded:

---

## 🧴 Skin Disease Classifier & OTC Medicine Guide

This project is a deep learning-based web tool that classifies skin diseases from images and suggests appropriate over-the-counter (OTC) medicines. It's built using **EfficientNetB3**, TensorFlow/Keras, and includes a clean inference pipeline for real-world use.

---

### 📌 Features

* 🔍 Classifies **8 skin diseases** using transfer learning
* 💊 Recommends OTC medicine for the detected disease
* 📈 Trained with callbacks: EarlyStopping, ReduceLROnPlateau
* 📊 Handles **class imbalance** and uses **image augmentation**
* 🧪 Supports real-time testing with sample images

---

### 📂 Project Structure

```
skin-disease-classifier/
│
├── skin-disease-classifier.ipynb  ← Main model building notebook
├── model/                         ← Trained model files (to be added)
├── data/                          ← Image dataset (local or Kaggle)
└── app/                           ← Flask app (if deployed)
```

---

### ⚙️ Tech Stack

* Python 3.x
* TensorFlow / Keras
* EfficientNetB3
* scikit-learn, NumPy, Matplotlib
* Google Colab / Jupyter Notebook

---

### 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/skin-disease-classifier.git
   cd skin-disease-classifier
   ```

2. Open the notebook:

   ```bash
   jupyter notebook skin-disease-classifier.ipynb
   ```

3. Run all cells to train, evaluate, and test the model.

---

### 📈 Results

* Accuracy: \~85–90% on validation data
* F1 Score, Precision & Recall metrics included in notebook

---

### 🩺 Supported Skin Conditions

* Acne
* Eczema
* Psoriasis
* Rosacea
* Ringworm
* Cellulitis
* Dermatitis
* Melanoma

---

### 💡 Future Improvements

* Add **Flask or Streamlit web app**
* Integrate **camera support** for real-time capture
* Extend to **multilingual** medicine recommendations
* Collect more diverse data for improved generalization

---

### 📜 License

MIT License

---

Let me know if you also want me to create a separate `requirements.txt` or add deployment instructions.
