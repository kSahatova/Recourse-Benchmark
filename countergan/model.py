import os
import json
import time
import pickle

import numpy as np
import pandas as pd
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, ActivityRegularization, Add, Dropout
from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical

from utils import compute_reconstruction_error


def generate_fake_samples(x, generator):
    """
    Use the input generator to generate samples.
    """
    return generator.predict(x)


def data_stream(x, y=None, batch_size=500):
    """
    Generate batches until exhaustion of the input data.
    """
    n_train = x.shape[0]
    if y is not None:
        assert n_train == len(y)
    n_complete_batches, leftover = divmod(n_train, batch_size)
    n_batches = n_complete_batches + bool(leftover)

    perm = np.random.permutation(n_train)
    for i in range(n_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        if y is not None:
            output = (x[batch_idx], y[batch_idx])
        else:
            output = x[batch_idx]
        yield output


def infinite_data_stream(x, y=None, batch_size=500):
    """Infinite batch generator."""
    batches = data_stream(x, y, batch_size=batch_size)
    while True:
        try:
            yield next(batches)
        except StopIteration:
            batches = data_stream(x, y, batch_size=batch_size)
            yield next(batches)


def create_generator(in_shape, residuals=True):  # =(X_train.shape[1],)
    """Define and compile the residual generator of the CounteRGAN."""
    generator_input = Input(shape=in_shape, name='generator_input')
    generator = Dense(64, activation='relu')(generator_input)
    generator = Dense(32, activation='relu')(generator)
    generator = Dense(64, activation='relu')(generator)
    generator = Dense(in_shape[0], activation='tanh')(generator)
    generator_output = ActivityRegularization(l1=0., l2=1e-6)(generator)

    if residuals:
        generator_output = Add(name="output")([generator_input, generator_output])

    return Model(inputs=generator_input, outputs=generator_output)


def create_discriminator(in_shape):
    """ Define a neural network binary classifier to classify real and generated
    examples."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=in_shape),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ], name="discriminator")
    optimizer = optimizers.Adam(lr=0.0005, beta_1=0.5, decay=1e-8)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy'])
    return model


def define_countergan(generator, discriminator, classifier,
                      input_shape):  # =(X_train.shape[1],)
    """Combine a generator, discriminator, and fixed classifier into the CounteRGAN."""
    discriminator.trainable = False
    classifier.trainable = False

    countergan_input = Input(shape=input_shape, name='countergan_input')

    x_generated = generator(countergan_input)

    countergan = Model(
        inputs=countergan_input,
        outputs=[discriminator(x_generated), classifier(x_generated)]
    )

    optimizer = optimizers.RMSprop(lr=2e-4, decay=1e-8)
    countergan.compile(optimizer, ["binary_crossentropy", "categorical_crossentropy"])
    return countergan


def define_weighted_countergan(generator, discriminator, classifier,
                               input_shape):
    """
    Combine a generator and a discriminator for the weighted version of the
    CounteRGAN.
    """
    discriminator.trainable = False
    classifier.trainable = False
    countergan_input = Input(shape=input_shape, name='countergan_input')

    x_generated = generator(countergan_input)

    countergan = Model(inputs=countergan_input, outputs=discriminator(x_generated))
    optimizer = optimizers.RMSprop(lr=5e-4, decay=1e-8)
    countergan.compile(optimizer, "binary_crossentropy")
    return countergan


def train_countergan(n_discriminator_steps, n_generator_steps,
                     n_training_iterations, desired_class, n_classes,
                     classifier, autoencoder, discriminator, generator,
                     batches, weighted_version=False):
    # todo : specify data normally
    """ Main function: train the CounteRGAN"""

    def check_divergence(x_generated):
        return np.all(np.isnan(x_generated))

    def print_training_information(generator, classifier, X_test, iteration):
        X_gen = generator.predict(X_test)
        clf_pred_test = classifier.predict(X_test)
        clf_pred = classifier.predict(X_gen)

        delta_clf_pred = (clf_pred - clf_pred_test)[:, desired_class]
        y_target = to_categorical([desired_class] * len(clf_pred),
                                  num_classes=n_classes)
        print('=' * 88)
        print(f"Training iteration {iteration} at {datetime.now()}")

        reconstruction_error = np.mean(compute_reconstruction_error(X_gen, autoencoder))
        print(f"Autoencoder reconstruction error (infinity to 0): {reconstruction_error:.3f}")
        print(f"Counterfactual prediction gain (0 to 1): {delta_clf_pred.mean():.3f}")
        print(f"Sparsity (L1, infinity to 0): {np.mean(np.abs(X_gen - X_test)):.3f}")

    if weighted_version:
        countergan = define_weighted_countergan(generator, discriminator,
                                                classifier, input_shape)
    else:
        countergan = define_countergan(generator, discriminator, classifier)

    discrim_loss_1, discrim_loss_2, discrim_accuracy, gan_loss, \
        discrim_loss, clf_loss = 0, 0, 0, 0, 0, 0
    for iteration in range(n_training_iterations):
        if iteration > 0:
            x_generated = generator.predict()
            if check_divergence(x_generated):
                print("Training diverged with the following loss functions:")
                # todo : never defined in the original code
                print(discrim_loss_1, discrim_accuracy, gan_loss,
                      discrim_loss, discrim_loss_2, clf_loss)
                break

        # Periodically print and plot training information
        if (iteration % 1000 == 0) or (iteration == n_training_iterations - 1):
            print_training_information(generator, classifier, X_test, iteration)

        # Train the discriminator
        discriminator.trainable = True
        for _ in range(n_discriminator_steps):
            x_fake_input, _ = next(batches)
            x_fake = generate_fake_samples(x_fake_input, generator)
            x_real = x_fake_input

            x_batch = np.concatenate([x_real, x_fake])
            y_batch = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_fake))])

            # Shuffle real and fake examples
            p = np.random.permutation(len(y_batch))
            x_batch, y_batch = x_batch[p], y_batch[p]

            if weighted_version:
                classifier_scores = classifier.predict(x_batch)[:, desired_class]

                # The following update to the classifier scores is needed to have the
                # same order of magnitude between real and generated samples losses
                real_samples = np.where(y_batch == 1.)
                average_score_real_samples = np.mean(classifier_scores[real_samples])
                classifier_scores[real_samples] /= average_score_real_samples

                fake_samples = np.where(y_batch == 0.)
                classifier_scores[fake_samples] = 1.

                discriminator.train_on_batch(
                    x_batch, y_batch, sample_weight=classifier_scores
                )
            else:
                discriminator.train_on_batch(x_batch, y_batch)

        # Train the generator
        discriminator.trainable = False
        for _ in range(n_generator_steps):
            x_fake_input, _ = next(batches)
            y_fake = np.ones(len(x_fake_input))
            if weighted_version:
                countergan.train_on_batch(x_fake_input, y_fake)
            else:
                y_target = to_categorical([desired_class] * len(x_fake_input),
                                          num_classes=n_classes)
                countergan.train_on_batch(x_fake_input, [y_fake, y_target])
    return countergan


