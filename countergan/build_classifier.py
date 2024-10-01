import os
import numpy as np
from typing import Union
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

tf.random.set_seed(2020)
np.random.seed(2020)


def create_classifier(input_shape):
    """Define and compile a neural network binary classifier."""
    model = Sequential([
        Dense(20, activation='relu', input_shape=input_shape),
        Dense(20, activation='relu'),
        Dense(2, activation='softmax'),
    ], name="classifier")
    optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy'])
    return model


def train_classifier(X_train, y_train, X_test, y_test,
                     weights_path_to_save: str) -> Union[Sequential, None]:
    """
    Training of the classifier
    :return:
    """
    cls_input = X_train.shape[1]
    classifier = create_classifier(input_shape=cls_input)

    training = classifier.fit(X_train, y_train, batch_size=32, epochs=200, verbose=0,
                              validation_data=(X_test, y_test),)
    print(f"Training: loss={training.history['loss'][-1]:.4f}, "
          f"accuracy={training.history['accuracy'][-1]:.4f}")
    print(f"Validation: loss={training.history['val_loss'][-1]:.4f}, "
          f"accuracy={training.history['val_accuracy'][-1]:.4f}")

    classifier.save(os.path.join(weights_path_to_save, 'classifier.h5'))

    return classifier
