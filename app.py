from flask import Flask, request, render_template, jsonify, url_for, send_file
import numpy as np
import tensorflow as tf
import cv2
import io
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.debug = True  # TEMP for tracing crashes

MODEL_PATH = 'model/best_model_efficientnet.keras'
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']

def load_model():
    print(f"[INFO] Attempting to load model from {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[SUCCESS] Model loaded.")
        return model
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load model: {e}")
        raise

@app.route('/debug_files')
def debug_files():
    # Lists model-related files in the project directory
    model_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.keras') or file.endswith('.h5'):
                model_files.append(os.path.join(root, file))
    print(f"[DEBUG] Found model files: {model_files}")
    return jsonify(model_files)

@app.route('/test_model')
def test_model():
    try:
        _ = load_model()
        return "Model loaded successfully!"
    except Exception as e:
        return f"Model failed to load: {e}", 500

@app.errorhandler(500)
def handle_internal_error(e):
    print(f"[ERROR 500] {e}")
    return jsonify({'error': 'Internal server error'}), 500

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
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        model = load_model()
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[class_index]
        confidence = float(predictions[0][class_index])

        return jsonify({'class': predicted_class, 'confidence': confidence})

    except Exception as e:
        print(f"[ERROR in /predict] {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gradcam_image', methods=['POST'])
def gradcam_image():
    try:
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resized = cv2.resize(orig, (224, 224))

        img_array = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))

        model = load_model()
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])

        base_model = model.get_layer('efficientnetb0')
        conv_layer = base_model.get_layer('top_conv')
        grad_model = tf.keras.models.Model(inputs=base_model.input, outputs=[conv_layer.output, base_model.output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        heatmap = np.uint8(255 * heatmap)
        cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        combined = cv2.addWeighted(resized, 0.6, cam, 0.4, 0)

        image = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"[ERROR in /gradcam_image] {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/saliency_image', methods=['POST'])
def saliency_image():
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        model = load_model()
        image_tensor = tf.Variable(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            preds = model(image_tensor, training=False)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, image_tensor)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
        saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
        saliency = np.uint8(255 * saliency.numpy())
        heatmap = cv2.applyColorMap(cv2.resize(saliency, (224, 224)), cv2.COLORMAP_HOT)
        image = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"[ERROR in /saliency_image] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
