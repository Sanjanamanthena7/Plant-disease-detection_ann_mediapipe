# import cv2

# from utils.predictor import predict_disease

# image = cv2.imread(
#     r"C:\Users\22053\Downloads\archive (2)\plantvillage dataset\color\Apple___Apple_scab\0a769a71-052a-4a18-8d18-b5c79f08355f___FREC_Scab 3165.JPG"
# )

# disease, confidence = predict_disease(image)

# print("Disease :", disease)
# print("Confidence :", confidence)

from tensorflow.keras.models import load_model

model = load_model("models/plant_disease_ann.h5")

print("Loaded Successfully")
