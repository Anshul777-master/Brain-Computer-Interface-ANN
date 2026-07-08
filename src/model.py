import os
import sys

# Add project root directory to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

import config



def create_ann(input_shape):

    model = Sequential()


    # Input Layer
    model.add(
        Input(
            shape=(input_shape,)
        )
    )


    # Hidden Layer 1
    model.add(
        Dense(
            128,
            activation="relu"
        )
    )


    model.add(
        Dropout(
            0.3
        )
    )


    # Hidden Layer 2
    model.add(
        Dense(
            64,
            activation="relu"
        )
    )


    model.add(
        Dropout(
            0.3
        )
    )


    # Hidden Layer 3
    model.add(
        Dense(
            32,
            activation="relu"
        )
    )


    # Output Layer
    # Binary Classification:
    # 0 = Low Valence
    # 1 = High Valence

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )


    # Optimizer

    optimizer = Adam(
        learning_rate=config.LEARNING_RATE
    )


    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model



# Test Model

if __name__ == "__main__":


    model = create_ann(352)


    model.summary()