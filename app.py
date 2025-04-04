from flask import Flask, request, render_template, jsonify, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Path to your trained model
MODEL_PATH = os.path.join('model', 'best_model_efficientnet.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']

@app.route('/')
def home():
    return render_template('home.html', background_image=url_for('static', filename='images/001.png'))

@app.route('/diagnostic')
def diagnostic():
    return render_template('diagnostic.html', background_image=url_for('static', filename='images/002.png'))

@app.route('/about')
def about():
    return render_template('about.html', background_image=url_for('static', filename='images/003.png'))

@app.route('/technical')
def technical():
    return render_template('technical.html', background_image=url_for('static', filename='images/004.png'))

@app.route('/research')
def research():
    return render_template('research.html', background_image=url_for('static', filename='images/005.png'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save the uploaded file temporarily
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Load and preprocess the image (resize only, no augmentations)
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)  # shape: (224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Normalize for EfficientNet

        # Predict with the model
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[class_index]
        confidence = float(predictions[0][class_index])

        # Remove the temporary file
        os.remove(filepath)

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)