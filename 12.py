import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to model and test image
MODEL_PATH = "model/best_model_efficientnet.keras"  # Update if needed
TEST_IMAGE_PATH = "test/adenocarcinoma.jpeg"  # Update this to any image in 'test/' folder

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# 🔥 **Force the model to be built by calling it once**
if model.inputs is None:
    print("⚠️ Model has no defined input. Building model with input shape (1, 224, 224, 3)...")
    model(tf.keras.Input(shape=(224, 224, 3)))  # ✅ Calls model once to define inputs

# 🔥 **Fix: Get the correct model input**
model_input = model.inputs[0]  # ✅ Fix: Directly use first input tensor
print("✅ Model input shape:", model_input.shape, "\n")

# Print model summary
print("\n===========================")
print("🔍 ANALYZING MODEL STRUCTURE")
print("===========================\n")
model.summary()

# Print all layers and their types
print("\n--- MODEL LAYERS ---")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} - {type(layer)}")
print("---------------------\n")

# Class names
CLASS_NAMES = ['lung_aca', 'lung_n', 'lung_scc']

def preprocess_image(image_path):
    """Loads and preprocesses the image for EfficientNet."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Convert to tensor explicitly
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return img_tensor

def generate_saliency_map(image_tensor, model):
    """Generates a Saliency Map based on the gradients."""
    image_tensor = tf.Variable(image_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, image_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
    
    return saliency.numpy(), class_idx.numpy()

def visualize_saliency(image_path, saliency_map, class_idx):
    """Displays Saliency Map results without saving."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Saliency Map
    ax[1].imshow(saliency_map, cmap='hot')
    ax[1].set_title(f"Saliency Map ({CLASS_NAMES[class_idx]})")
    ax[1].axis("off")

    plt.show()

if __name__ == "__main__":
    print("Loading and processing image...")
    img_tensor = preprocess_image(TEST_IMAGE_PATH)

    print("Generating Saliency Map...")
    saliency_map, class_idx = generate_saliency_map(img_tensor, model)

    print("Visualizing results...")
    visualize_saliency(TEST_IMAGE_PATH, saliency_map, class_idx)
