import json
import cv2
import numpy as np
from keras.models import load_model

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = r"C:\Users\22053\OneDrive\Desktop\plant_ann\models\plant_disease_ann.keras"

model = load_model(MODEL_PATH, compile=False)
CLASS_PATH = "models/class_names.json"

model = load_model(MODEL_PATH, compile=False)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = 64

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0

    image = image.flatten()

    image = np.expand_dims(image, axis=0)

    return image


# ==========================================================
# SPLIT CLASS NAME
# ==========================================================

def split_class(class_name):

    if "___" in class_name:

        plant, disease = class_name.split("___")

    else:

        plant = class_name
        disease = class_name

    disease = disease.replace("_", " ")
    plant = plant.replace("_", " ")

    return plant, disease


# ==========================================================
# PREDICT DISEASE
# ==========================================================

def predict_disease(image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)[0]

    index = np.argmax(prediction)

    confidence = float(prediction[index])

    class_name = class_names[index]

    plant, disease = split_class(class_name)

    healthy = "healthy" in disease.lower()

    return {

        "plant": plant,

        "disease": disease,

        "healthy": healthy,

        "confidence": confidence

    }


# ==========================================================
# TOP 5 PREDICTIONS
# ==========================================================

def top_predictions(image, k=5):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)[0]

    indices = np.argsort(prediction)[::-1][:k]

    results = []

    for idx in indices:

        plant, disease = split_class(class_names[idx])

        results.append({

            "plant": plant,

            "disease": disease,

            "confidence": float(prediction[idx])

        })

    return results