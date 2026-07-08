import os
import pickle
import numpy as np
import config

# ==========================
# Dataset Configuration
# ==========================

DATASET_PATH = "dataset"

# Emotion to classify
# Options:
# valence
# arousal
# dominance
# liking

TARGET_EMOTION = "valence"

# Threshold for binary classification
THRESHOLD = 5.0

# EEG Settings

NUM_CHANNELS = 32
SAMPLING_RATE = 128

# Model Settings

TEST_SIZE = 0.2

RANDOM_STATE = 42

EPOCHS = 50

BATCH_SIZE = 32

LEARNING_RATE = 0.001

MODEL_PATH = "models/emotion_ann.keras"

OUTPUT_PATH = "outputs"

os.makedirs(MODEL_PATH.rsplit("/",1)[0],exist_ok=True)
os.makedirs(OUTPUT_PATH,exist_ok=True)