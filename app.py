import streamlit as st
import cv2
import numpy as np
import pandas as pd

from PIL import Image

import sys
import tensorflow as tf
from tensorflow import keras

try:
    import av
except:
    av = None

# print("="*60)
# print(sys.executable)
# print("TensorFlow:", tf.__version__)
# print("Keras:", keras.__version__)
# print("="*60)

# from streamlit_webrtc import (
#     webrtc_streamer,
#     VideoProcessorBase
# )


from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

    

from utils.predictor import (
    predict_disease,
    top_predictions
)

from utils.disease_info import (
    get_disease_info
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""

<style>

.main{
    background:#f8fff8;
}

.block-container{
    padding-top:2rem;
}

.title{

    font-size:42px;

    font-weight:bold;

    color:#228B22;

    text-align:center;

}

.subtitle{

    font-size:18px;

    color:gray;

    text-align:center;

}

.card{

    background:white;

    padding:18px;

    border-radius:15px;

    border:1px solid #dddddd;

    box-shadow:0px 2px 10px rgba(0,0,0,0.10);

}

.footer{

    text-align:center;

    color:gray;

    font-size:14px;

}

.disease{

    color:red;

    font-weight:bold;

}

.healthy{

    color:green;

    font-weight:bold;

}

</style>

""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(

    "<div class='title'>🌿 Plant Disease Detection using ANN</div>",

    unsafe_allow_html=True

)

st.markdown(

    "<div class='subtitle'>Artificial Neural Network | PlantVillage Dataset</div>",

    unsafe_allow_html=True

)

st.markdown("---")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌿 Project")

st.sidebar.success("Artificial Neural Network")

st.sidebar.info("PlantVillage Dataset")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.metric(

    "Algorithm",

    "ANN"

)

st.sidebar.metric(

    "Input Size",

    "64 x 64"

)

st.sidebar.metric(

    "Classes",

    "38"

)

st.sidebar.markdown("---")

st.sidebar.subheader("Supported Plants")

plants = [

    "🍎 Apple",

    "🫐 Blueberry",

    "🍒 Cherry",

    "🌽 Corn",

    "🍇 Grape",

    "🍊 Orange",

    "🍑 Peach",

    "🫑 Pepper",

    "🥔 Potato",

    "🍓 Strawberry",

    "🍅 Tomato"

]

for p in plants:

    st.sidebar.write(p)

st.sidebar.markdown("---")

mode = st.sidebar.radio(

    "Select Mode",

    [

        "📤 Upload Image",

        "📹 Live Webcam"

    ]

)

st.markdown("---")

st.subheader(mode)


# ==========================================================
# UPLOAD IMAGE
# ==========================================================

if mode == "📤 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = predict_disease(image)

        predictions = top_predictions(image)

        plant = result["plant"]

        disease = result["disease"]

        confidence = result["confidence"]

        healthy = result["healthy"]

        class_name = plant.replace(" ", "_") + "___" + disease.replace(" ", "_")

        info = get_disease_info(class_name)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])

        # ==========================================================
        # IMAGE
        # ==========================================================

        with col1:

            st.image(
                rgb,
                # use_container_width=False
            )

        # ==========================================================
        # RESULT CARD
        # ==========================================================

        with col2:

            st.success("Prediction")

            st.metric(
                "Plant",
                plant
            )

            st.metric(
                "Disease",
                disease
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(float(confidence))

            if healthy:

                st.success("🌿 Healthy Plant")

            else:

                st.error("⚠ Disease Detected")

        st.markdown("---")

        # ==========================================================
        # DISEASE DESCRIPTION
        # ==========================================================

        st.subheader("📖 Disease Information")

        st.info(info["description"])

        # ==========================================================
        # SYMPTOMS
        # ==========================================================

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("🩺 Symptoms")

            for symptom in info["symptoms"]:

                st.write("✅", symptom)

        # ==========================================================
        # TREATMENT
        # ==========================================================

        with c2:

            st.subheader("💊 Treatment")

            for treat in info["treatment"]:

                st.write("🌱", treat)

        st.markdown("---")

        # ==========================================================
        # TOP 5 PREDICTIONS
        # ==========================================================

        st.subheader("📊 Top 5 Predictions")

        table = []

        for pred in predictions:

            table.append({

                "Plant": pred["plant"],

                "Disease": pred["disease"],

                "Confidence (%)": round(pred["confidence"]*100,2)

            })

        df = pd.DataFrame(table)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            df.set_index("Disease")["Confidence (%)"]
        )


# ==========================================================
# LIVE WEBCAM
# ==========================================================
# if webcam_available:

class PlantDiseaseProcessor(VideoProcessorBase):

   def recv(self, frame):   

        image = frame.to_ndarray(format="bgr24")

        output = image.copy()

        try:

            result = predict_disease(image)

            plant = result["plant"]

            disease = result["disease"]

            confidence = result["confidence"]

            healthy = result["healthy"]

            color = (0,255,0)

            if not healthy:

                color = (0,0,255)

            h, w = output.shape[:2]

            cv2.rectangle(
                output,
                (20,20),
                (w-20,180),
                color,
                3
            )

            cv2.putText(
                output,
                f"Plant : {plant}",
                (40,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                output,
                f"Disease : {disease}",
                (40,95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                output,
                f"Confidence : {confidence*100:.2f}%",
                (40,130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            status = "Healthy"

            if not healthy:

                status = "Disease Detected"

            cv2.putText(
                output,
                f"Status : {status}",
                (40,165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except:

            pass

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )



# ==========================================================
# LIVE CAMERA MODE
# ==========================================================

if mode == "📹 Live Webcam":

#if webcam_available:

    st.info("Click START and show a leaf.")

    webrtc_streamer(
        key="plant-disease",
        video_processor_factory=PlantDiseaseProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

# else:

#     st.warning("Webcam is not supported on this deployment.")

st.markdown("---")

st.subheader("Supported Crops")

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.success("🍎 Apple")

    st.success("🍅 Tomato")

    st.success("🥔 Potato")

with c2:

    st.success("🌽 Corn")

    st.success("🍇 Grape")

    st.success("🍑 Peach")

with c3:

    st.success("🍊 Orange")

    st.success("🍓 Strawberry")

    st.success("🫑 Pepper")

with c4:

    st.success("🍒 Cherry")

    st.success("🫐 Blueberry")

    st.success("🌱 Soybean")



# ==========================================================
# PROJECT DASHBOARD
# ==========================================================

st.markdown("---")

st.subheader("📊 Project Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Model",
        "ANN"
    )

with c2:
    st.metric(
        "Dataset",
        "PlantVillage"
    )

with c3:
    st.metric(
        "Classes",
        "38"
    )

with c4:
    st.metric(
        "Image Size",
        "64 × 64"
    )

st.markdown("---")

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.subheader("ℹ About this Project")

st.write("""
This application detects diseases in plant leaves using an
**Artificial Neural Network (ANN)** trained on the
**PlantVillage Dataset**.

### Features

- 🌿 Detect plant diseases from uploaded images
- 📹 Live webcam disease prediction
- 🤖 ANN-based classification
- 📊 Confidence score
- 📈 Top-5 predictions
- 📖 Disease description
- 💊 Treatment suggestions

""")

st.info(
"""
⚠ **Note**

For the best prediction results:

• Upload a clear image of a single leaf.

• Avoid blurry or dark images.

• During live webcam prediction, keep the leaf close to the camera and use a plain background.
"""
)

st.markdown("---")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div class='footer'>

🌿 <b>Plant Disease Detection using Artificial Neural Network</b><br><br>

Dataset : <b>PlantVillage</b><br>

Developed using <b>Python • TensorFlow • OpenCV • Streamlit</b>

</div>
""",
unsafe_allow_html=True
)
