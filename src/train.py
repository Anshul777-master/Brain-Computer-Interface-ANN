import os
import sys

# Add project root path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib

from model import create_ann
import config



# ==============================
# Load Data
# ==============================

X = np.load(
    "outputs/features.npy"
)

labels = np.load(
    "outputs/labels.npy"
)


print("Feature Shape:", X.shape)

print("Labels Shape:", labels.shape)



# ==============================
# Select Valence Labels
# ==============================

valence = labels[:,0]


# Convert into binary classes

y = np.where(
    valence >= config.THRESHOLD,
    1,
    0
)


print("\nClass Distribution:")

print(
    np.unique(
        y,
        return_counts=True
    )
)



# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=config.TEST_SIZE,

    random_state=config.RANDOM_STATE,

    stratify=y
)



print("\nTraining Samples:", X_train.shape)

print("Testing Samples:", X_test.shape)



# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



joblib.dump(
    scaler,
    "models/scaler.pkl"
)



# ==============================
# Create ANN
# ==============================

model = create_ann(
    X_train.shape[1]
)



model.summary()



# ==============================
# Train Model
# ==============================

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=config.EPOCHS,

    batch_size=config.BATCH_SIZE,

    verbose=1

)



# ==============================
# Save Model
# ==============================

model.save(
    config.MODEL_PATH
)


print("\nModel Saved Successfully")



# ==============================
# Save Accuracy Graph
# ==============================

plt.figure(figsize=(8,5))


plt.plot(
    history.history["accuracy"]
)


plt.plot(
    history.history["val_accuracy"]
)


plt.title(
    "ANN Training Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend(
    [
        "Training",
        "Validation"
    ]
)


plt.savefig(
    "outputs/training_accuracy.png"
)


plt.close()



print("Training Graph Saved")