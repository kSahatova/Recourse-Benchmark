import os
import json
import numpy as np
from pprint import pprint
from typing import Union

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Add, Input, ActivityRegularization
from tensorflow.keras import Model, optimizers, regularizers
from tensorflow.keras.utils import to_categorical

from utils import compute_reconstruction_error, format_metric

tf.random.set_seed(2020)
np.random.seed(2020)


def add_noise(x, noise_factor=1e-6):
    """
    Adds noise to the input data x
    :param x:
    :param noise_factor:
    :return:
    """
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    return x_noisy


def create_autoencoder(in_shape) -> Model:
    """
    Builds an autoencoder
    :param in_shape: input shape to construct AE
    :return: tensorflow.keras.Model of autoencoder
    """
    input_ = Input(shape=in_shape)

    x = Dense(32, activation="relu")(input_)
    encoded = Dense(8)(x)
    x = Dense(32, activation="relu")(encoded)
    decoded = Dense(in_shape[0], activation="tanh")(x)

    model = Model(input_, decoded)
    optimizer = optimizers.Nadam()
    model.compile(optimizer, 'mse')
    return model


def train_autoencoder(X_train, X_test, weights_path_to_save: str) -> None:
    """
    Function to the training of the AE
    :param X_train:
    :param X_test:
    :param weights_path_to_save:
    :return:
    """
    ae_input = X_train.shape[1]
    autoencoder = create_autoencoder(ae_input)
    training = autoencoder.fit(
        add_noise(X_train), X_train, epochs=100, batch_size=32, shuffle=True,
        validation_data=(X_test, X_test), verbose=0
    )
    print(f"Training loss: {training.history['loss'][-1]:.4f}")
    print(f"Validation loss: {training.history['val_loss'][-1]:.4f}")

    n_samples = 1000
    # Compute the reconstruction error of noise data
    samples = np.random.randn(n_samples, X_train.shape[1])
    reconstruction_error_noise = compute_reconstruction_error(samples, autoencoder)

    # Save and print the autoencoder metrics
    reconstruction_error = compute_reconstruction_error(X_test, autoencoder)
    autoencoder_metrics = {
        "reconstruction_error": format_metric(reconstruction_error),
        "reconstruction_error_noise": format_metric(reconstruction_error_noise),
    }
    pprint(autoencoder_metrics)
    autoencoder.save(os.path.join(weights_path_to_save, 'autoencoder.h5'))



if __name__ == '__main__':
    train_autoencoder()
