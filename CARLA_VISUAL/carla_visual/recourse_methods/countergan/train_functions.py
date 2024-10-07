import os
import torch 
import numpy as np
from torch import optim
from pprint import pprint

from typing import Union
from tensorflow.keras import models
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential

from .utils import compute_reconstruction_error, format_metric


def train_classifier(model: Sequential,
                     train_dataset, 
                     val_dataset, 
                     epochs: int,
                     name: str,
                     weights_path_to_save: str) -> Union[models.Sequential, None]:
    """
    Training of the classifier
    :return:
    """

    training = model.fit(train_dataset, 
                         epochs=epochs, 
                         verbose=1,
                         validation_data=val_dataset)
    
    print(f"Training: loss={training.history['loss'][-1]:.4f}, "
          f"accuracy={training.history['accuracy'][-1]:.4f}")
    print(f"Validation: loss={training.history['val_loss'][-1]:.4f}, "
          f"accuracy={training.history['val_accuracy'][-1]:.4f}")

    model.save(os.path.join(weights_path_to_save, f'{name}.h5'))

    return model


def add_noise(x, noise_factor=1e-6):
    """
    Adds noise to the input data x
    :param x:
    :param noise_factor:
    :return:
    """
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    return x_noisy


def train_autoencoder(autoencoder, X_train, X_test, weights_path_to_save: str) -> None:
    """
    Function to the training of the AE
    :param X_train:
    :param X_test:
    :param weights_path_to_save:
    :return:
    """
    # ae_input = X_train.shape[1]
    #autoencoder = autoencoder(ae_input)
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
