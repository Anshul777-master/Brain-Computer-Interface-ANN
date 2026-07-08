import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import numpy as np
import joblib

from tensorflow.keras.models import load_model

import config



# ==========================
# Load Model
# ==========================

model = load_model(
    config.MODEL_PATH
)


# Load scaler

scaler = joblib.load(
    "models/scaler.pkl"
)



# ==========================
# Prediction Function
# ==========================

def predict_emotion(features):


    # Convert input to numpy array

    features = np.array(
        features
    )


    # Reshape

    features = features.reshape(
        1,
        -1
    )


    # Scale

    features = scaler.transform(
        features
    )


    # Predict probability

    prediction = model.predict(
        features
    )


    probability = prediction[0][0]


    if probability >= 0.5:

        result = "High Valence (Positive Emotion)"

    else:

        result = "Low Valence (Negative Emotion)"



    return result, probability





# ==========================
# Test Prediction
# ==========================

if __name__ == "__main__":


    # Load random EEG feature sample

    X = np.load(
        "outputs/features.npy"
    )


    sample = X[0]


    result, confidence = predict_emotion(
        sample
    )


    print("\nEEG Emotion Prediction")
    print("-----------------------")


    print(
        "Prediction:",
        result
    )


    print(
        "Confidence:",
        round(
            float(confidence)*100,
            2
        ),
        "%"
    )