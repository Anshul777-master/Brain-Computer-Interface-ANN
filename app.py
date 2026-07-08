import os
import sys

# Add project root
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# ==========================
# Configuration
# ==========================

MODEL_PATH = "models/emotion_ann.keras"

SCALER_PATH = "models/scaler.pkl"



# ==========================
# Load Model
# ==========================

@st.cache_resource
def load_ai_model():

    model = load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler



model, scaler = load_ai_model()



# ==========================
# UI Design
# ==========================

st.title(
    "🧠 Brain Computer Interface"
)


st.subheader(
    "EEG Signal Mental State Classification using ANN"
)


st.write(
    """
This application uses an Artificial Neural Network
trained on the DEAP EEG dataset to classify
brain activity into emotional states.
"""
)



# ==========================
# Upload Feature File
# ==========================


uploaded_file = st.file_uploader(
    "Upload EEG Feature File (.npy)",
    type=["npy"]
)



if uploaded_file:


    features = np.load(
        uploaded_file
    )


    st.success(
        "EEG Features Loaded Successfully"
    )


    st.write(
        "Feature Shape:",
        features.shape
    )


    # If multiple samples uploaded
    if len(features.shape) > 1:

        features = features[0]


    features = features.reshape(
        1,
        -1
    )


    # Scaling

    features = scaler.transform(
        features
    )


    if st.button(
        "Predict Mental State"
    ):


        prediction = model.predict(
            features
        )


        confidence = float(
            prediction[0][0]
        )


        if confidence >= 0.5:

            result = "😊 Positive Mental State (High Valence)"

        else:

            result = "😔 Negative Mental State (Low Valence)"



        st.divider()


        st.header(
            "Prediction Result"
        )


        st.success(
            result
        )


        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )



else:

    st.warning(
        "Please upload an EEG feature file"
    )