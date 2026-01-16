import os
from tensorflow.keras.models import load_model

from pathlib import Path

# Dictionary to hold all loaded models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "eye_efficientnet_v2_b3.keras")

model = {}

def load_models():
    """
    Load all EfficientNet-B3 models into memory at startup.
    """
    if not model:
        model["eye"] = load_model(MODEL_PATH, compile=False)

        print("EfficientNetV2-B3 models loaded Successfully.")

def get_model(task: str):
    return model.get(task)