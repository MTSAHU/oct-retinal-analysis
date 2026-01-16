from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import numpy as np

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((300, 300))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image