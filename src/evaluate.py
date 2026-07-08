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

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

import config



# =========================
# Load Data
# =========================

X = np.load(
    "outputs/features.npy"
)


labels = np.load(
    "outputs/labels.npy"
)



# =========================
# Prepare Labels
# =========================

valence = labels[:,0]


y = np.where(
    valence >= config.THRESHOLD,
    1,
    0
)



# =========================
# Train Test Split
# Same as training
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=config.TEST_SIZE,

    random_state=config.RANDOM_STATE,

    stratify=y
)



# =========================
# Load Scaler
# =========================

scaler = joblib.load(
    "models/scaler.pkl"
)


X_test = scaler.transform(
    X_test
)



# =========================
# Load Model
# =========================

model = load_model(
    config.MODEL_PATH
)



# =========================
# Prediction
# =========================

prediction = model.predict(
    X_test
)


y_pred = np.where(
    prediction >= 0.5,
    1,
    0
)



# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)



print("\nModel Performance")

print("----------------------")

print(
    "Accuracy:",
    accuracy
)


print(
    "Precision:",
    precision
)


print(
    "Recall:",
    recall
)


print(
    "F1 Score:",
    f1
)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(5,4)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "EEG Emotion Classification"
)


plt.savefig(
    "outputs/confusion_matrix.png"
)


plt.close()



print(
    "\nConfusion Matrix Saved"
)