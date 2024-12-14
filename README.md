# Lung Cancer Detection System

The **Lung Cancer Detection System** is an AI-powered application designed to classify histopathological images of lung tissue into three categories: benign lung tissue, lung adenocarcinoma, and lung squamous cell carcinoma. This project leverages deep learning through Convolutional Neural Networks (CNNs) and provides an intuitive web-based interface for easy use by medical professionals.

---

## Features

- **Accurate Image Classification**: Achieves over 94% accuracy in classifying histopathological images.
- **Web Application Interface**: Upload images and receive classification results with probability metrics.
- **Educational Value**: Suitable for research, clinical support, and training purposes.
- **Scalable Architecture**: Built with Flask and TensorFlow for ease of deployment.

---

## Installation Guide

### 1. Clone the Repository
```bash
# Clone the repository to your local machine
git clone https://github.com/liviuorehovschi/capstone.git
cd capstone
```

### 2. Create and Activate a Virtual Environment
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (Linux/MacOS)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Run the Application
```bash
# Start the Flask app
python app.py
```

The application will run locally, and you can access it in your browser at `http://127.0.0.1:5000`.

---

## Upload Images for Testing

To test the diagnostic tool, you can upload images from the `test_images` directory:

1. Navigate to the `Diagnostic` page in the web application.
2. Select any image from the `test` directory (provided in this repository).
3. Upload the image to receive predictions.

---

## Dataset
The dataset used for training the model is the **Lung and Colon Cancer Histopathological Image Dataset (LC25000)**, which includes:

- **Lung Adenocarcinoma**
- **Lung Squamous Cell Carcinoma**
- **Benign Lung Tissue**

For more details on the dataset, refer to the [LC25000 dataset paper](https://doi.org/10.48550/arXiv.1912.12142).

---

## Project Structure

```
capstone/
├── app.py                  # Main application file
├── static/                 # Static files (CSS, JavaScript, images)
├── templates/              # HTML templates for the web app
├── model/                  # Pre-trained CNN model
├── test/                   # Example images for testing
├── requirements.txt        # Python dependencies
└── README.md               # Project README file
```

---

## Contributions
Contributions to this project are welcome! If you find a bug or have suggestions for new features, feel free to open an issue or submit a pull request.


---

## Acknowledgments
- Dataset by Andrew A. Borkowski et al.
- EfficientNet model implementation by Tan & Le.

---

## Contact
If you have any questions, feel free to reach out:
- **Author**: Liviu Orehovschi
