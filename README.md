# 🌿 Plant Disease Detection using Artificial Neural Network (ANN)

## 📌 Project Overview

This project is a Plant Disease Detection System developed using an **Artificial Neural Network (ANN)**. It classifies plant leaf images into one of **38 disease classes** from the PlantVillage dataset.

The application provides two prediction modes:

- 📁 Upload Leaf Image
- 📷 Live Webcam Detection

The system predicts:

- Plant Name
- Disease Name
- Confidence Score
- Plant Health Status (Healthy/Diseased)

---

## 🎯 Objectives

- Detect diseases from plant leaf images.
- Build a machine learning model using ANN.
- Provide real-time disease prediction.
- Create a user-friendly Streamlit web application.

---

## 📂 Dataset

Dataset Used:
**PlantVillage Dataset**

Features:

- 38 Classes
- Healthy and Diseased leaves
- Multiple crop types

Examples:

- Apple___Black_rot
- Tomato___Early_blight
- Potato___Healthy
- Grape___Esca

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Streamlit
- JSON

---

## 📁 Project Structure

```
plant_ann/
│
├── dataset/
│   └── plant_disease_dataset.csv
│
├── models/
│   ├── plant_disease_ann.keras
│   ├── plant_disease_ann.h5
│   └── class_names.json
│
├── notebooks/
│   └── train_model.ipynb
│
├── utils/
│   └── predictor.py
│
├── app.py
├── create_csv.py
├── requirements.txt
└── README.md
```

---

## 🔄 Project Workflow

```
PlantVillage Dataset
        │
        ▼
Load Images
        │
        ▼
Resize Images (64×64)
        │
        ▼
Normalize Pixel Values
        │
        ▼
Flatten Images
        │
        ▼
Train ANN Model
        │
        ▼
Save Model
        │
        ▼
Streamlit Application
        │
        ├── Upload Image
        └── Webcam
                │
                ▼
Prediction
                │
                ▼
Display Result
```

---

## 🧠 ANN Architecture

```
Input Layer
12288 Neurons

↓

Dense (512)
ReLU

↓

Dropout (0.3)

↓

Dense (256)
ReLU

↓

Dropout (0.3)

↓

Dense (128)
ReLU

↓

Output Layer
38 Neurons
Softmax
```

---

## 🖼 Image Preprocessing

Each image undergoes the following preprocessing steps:

1. Read image using OpenCV
2. Resize image to 64×64 pixels
3. Normalize pixel values (0–255 → 0–1)
4. Flatten image into a 12288-element vector
5. Feed into ANN model

---

## 🎓 Training Process

- Load dataset
- Read images
- Resize images
- Normalize pixel values
- Flatten images
- Encode labels
- Split into training and testing sets
- Train ANN model
- Evaluate accuracy
- Save trained model

---

## 📈 Prediction Process

When a user uploads an image or uses the webcam:

```
Input Image

↓

Resize

↓

Normalize

↓

Flatten

↓

ANN Model

↓

Softmax Prediction

↓

Highest Probability

↓

Plant Name
Disease Name
Confidence Score
Health Status
```

---

## 💻 Application Features

### 📁 Upload Image

- Upload JPG, JPEG, or PNG images.
- Predict plant disease instantly.
- Display confidence score.

### 📷 Live Webcam

- Real-time prediction using webcam.
- Continuous disease detection.
- Displays prediction directly on the video feed.

---

## 📊 Output

The application displays:

- 🌱 Plant Name
- 🦠 Disease Name
- 📈 Confidence Score
- ✅ Healthy / ❌ Diseased Status

---

## ▶️ How to Run

### Step 1

Clone the repository

```bash
git clone <repository-link>
```

### Step 2

Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3

Run the application

```bash
streamlit run app.py
```

---

## 📸 Sample Output

### Upload Image

- Plant: Blueberry
- Disease: Healthy
- Confidence: 93.84%

### Live Webcam

- Real-time disease prediction
- Confidence displayed on the webcam screen

---

## 📁 Saved Model Files

| File | Description |
|------|-------------|
| plant_disease_ann.keras | Trained ANN Model |
| plant_disease_ann.h5 | Backup Model |
| class_names.json | Disease Class Labels |

---

## 🚀 Future Enhancements

- Improve model accuracy
- Mobile application support
- Cloud deployment
- Automatic treatment recommendations
- Support for additional plant species

---

## 📚 Conclusion

This project demonstrates how an Artificial Neural Network (ANN) can be used for plant disease classification. The model processes preprocessed leaf images and predicts the corresponding disease class. A Streamlit-based interface enables users to perform predictions using either uploaded images or a live webcam.

---

## 👨‍💻 Author

**Name:** *Sanjana Manthena*

**Project Title:** Plant Disease Detection using Artificial Neural Network (ANN)

**Academic Project**