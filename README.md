# 🧬 Lung Cancer Histopathology Classifier (Capstone Project)

This is a Flask-based web application that assists users in diagnosing lung cancer subtypes from histopathological images. The model classifies each uploaded image into one of three categories:

- **Benign Lung Tissue**
- **Adenocarcinoma**
- **Squamous Cell Carcinoma**

After generating a prediction, the app displays a **confidence score** and an **interpretability message** to help users understand the certainty behind the model's output. Interpretations range from “Very high confidence” to “Lower confidence,” guiding the user on how much trust to place in the result and whether further testing is advised.

Users can also generate:
- ✅ A **Saliency Map** to see which parts of the image influenced the model
- ✅ A **Grad-CAM Heatmap** for deeper visual interpretability

> ⚠️ **Disclaimer**: This is a research and educational tool. It is **not a clinical diagnostic tool** and should not be used as a substitute for professional medical diagnosis.

---

## 🚀 Features

- Upload high-resolution histopathological images
- Predict lung tissue class using a deep learning model
- Display confidence scores and interpretability feedback
- Generate visual explanations via Saliency Maps and Grad-CAM
- Built with **EfficientNetB0**, **TensorFlow**, **OpenCV**, and **Flask**

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/liviuorehovschi/capstone.git
cd capstone
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

### 4. Install the Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

After installing everything and activating the virtual environment, start the Flask development server:

```bash
flask run
```

Then open your browser and go to:

```text
http://127.0.0.1:5000/
```

You’ll be able to:

- Upload an image  
- Get the predicted diagnosis with a confidence percentage and interpretation  
- Generate a Saliency Map and Grad-CAM visualization  

---

## 🧠 Model Info

- **Model Architecture**: EfficientNetB0 (transfer learning)
- **Training Data**: Combined dataset of LC25000 and LungHist700
- **Accuracy**: 98.25% on test set
- **Prediction Method**: Softmax probabilities
- **Explainability**: Saliency mapping and Grad-CAM overlays

---

## 🗂 Project Structure

```bash
capstone/
│
├── app.py              # Main Flask app
├── model/              # Trained CNN model
├── static/             # CSS, JS, and images
├── templates/          # HTML templates
├── test/               # Sample histopathology images
├── requirements.txt    # Project dependencies
├── .gitignore          # Excludes venv, pycache, etc.
└── README.md           # This file
```

---

## 📊 Confidence Interpretation Scale

After prediction, the app also interprets the model's confidence to guide user understanding:

- **≥ 95%**: Very high confidence – highly reliable prediction  
- **85–94.99%**: High confidence – likely correct, clinical follow-up recommended  
- **70–84.99%**: Moderate confidence – plausible prediction, further testing advised  
- **< 70%**: Lower confidence – preliminary result, clinical verification required  

## 🙌 Acknowledgments

- LC25000 and LungHist700 datasets  
- EfficientNetB0 by Tan & Le  
- TensorFlow, Flask, OpenCV communities  
- All contributors to open-source tools used in this project

---

## ✍️ Author

**Liviu Orehovschi**  
GitHub: [@liviuorehovschi](https://github.com/liviuorehovschi)  
LinkedIn: [Liviu Orehovschi](https://www.linkedin.com/in/liviuorehovschi)
