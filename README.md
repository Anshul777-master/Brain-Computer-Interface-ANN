# 🧠 Brain Computer Interface using EEG Signals and ANN

## Overview

This project implements a Brain Computer Interface (BCI) system that analyzes EEG signals and classifies emotional states using Artificial Neural Networks.

The system uses EEG signals from the DEAP dataset and performs preprocessing, feature extraction, ANN training, and emotion prediction.

---

## Features

- EEG signal preprocessing
- Bandpass filtering
- EEG feature extraction
- Artificial Neural Network classifier
- Emotion classification
- Model evaluation
- Streamlit web application

---

## Architecture
EEG Signals
|
↓
Preprocessing
|
↓
Feature Extraction
|
↓
ANN Model
|
↓
Emotion Prediction

---

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- SciPy
- Scikit-learn
- MNE
- Streamlit

---

## Project Structure
Brain_Computer_Interface_ANN/

├── src/
├── models/
├── outputs/
├── dataset/
├── app.py
├── config.py
└── requirements.txt

## Installation

Clone repository:

bash
git clone <repository-url>

Install dependencies:

pip install -r requirements.txt
Dataset

This project uses the DEAP EEG dataset.

Download the dataset separately and place .dat files inside:

dataset/
Run Project

Train model:
python src/train.py

Evaluate:

python src/evaluate.py

Run application:

streamlit run app.py
Model

ANN Architecture:

Dense Layer: 128 neurons
Dense Layer: 64 neurons
Dense Layer: 32 neurons
Output Layer: Sigmoid
Author

Anshul Gupta