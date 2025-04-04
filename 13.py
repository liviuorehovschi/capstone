import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
MODEL_PATH = "model/best_model_efficientnet.keras"
TEST_IMAGE_PATH = "test/adenocarcinoma.jpeg"
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']

# Load model
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure model input is defined
if model.inputs is None:
    print("⚠️ Model input is not defined. Forcing input definition.")
    model(tf.keras.Input(shape=(224, 224, 3)))  # Match the input shape of your model

def preprocess_image(image_path):
    """Preprocesses image for EfficientNet."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def get_last_conv_layer(model):
    """Recursively searches for the last Conv2D layer inside model or its submodules."""
    def find_conv_layers(m):
        conv_layers = []
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
            elif hasattr(layer, 'layers'):  # nested model
                conv_layers.extend(find_conv_layers(layer))
        return conv_layers

    conv_layers = find_conv_layers(model)
    
    if not conv_layers:
        print("\n❌ Still couldn't find any Conv2D layers. Your model structure may be too custom.")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} - {type(layer)}")
        raise ValueError("No Conv2D layer found anywhere in the model.")
    
    last_conv = conv_layers[-1]
    print(f"✅ Found Conv2D layer: {last_conv}")
    return last_conv
def generate_gradcam(model, image, conv_layer_name, class_index):
    """
    Generate Grad-CAM heatmap using inner EfficientNet model.
    """
    # 1. Get the base EfficientNet model
    base_model = model.get_layer('efficientnetb0')
    
    # 2. Build the grad model with base_model.input (which is defined!)
    target_layer = base_model.get_layer(conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )

    # 3. Run Grad-CAM
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()




def overlay_gradcam(original_img_path, heatmap, alpha=0.4):
    """Overlays the heatmap on original image."""
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img[..., ::-1]  # Convert BGR to RGB

if __name__ == "__main__":
    print("📸 Preprocessing image...")
    img_array = preprocess_image(TEST_IMAGE_PATH)

    print("🤖 Running model prediction...")
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_name = CLASS_NAMES[class_idx]
    confidence = preds[0][class_idx]
    print(f"🔬 Predicted class: {class_name} ({confidence:.2f})")

    print("🔍 Searching for last Conv2D layer...")
    conv_layer_name = get_last_conv_layer(model)

    print("🧠 Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam(model, img_array, conv_layer_name, class_idx)

    print("🖼️ Overlaying heatmap...")
    result_img = overlay_gradcam(TEST_IMAGE_PATH, heatmap)

    print("📊 Displaying results...")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(TEST_IMAGE_PATH)[..., ::-1])
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.title(f"Grad-CAM: {class_name}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
