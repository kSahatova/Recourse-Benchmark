import os
import torch 
import numpy as np
from torch import optim
from pprint import pprint

from typing import Union
from tensorflow.keras import models
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from carla_visual.plotting.plot_output import plot_generated_images


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

    model.save(os.path.join(weights_path_to_save, f'{name}.keras'))

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


def train_autoencoder(autoencoder, X_train, X_test, epochs,
                      weights_path_to_save: str) -> None:
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
        add_noise(X_train), X_train, epochs=epochs, batch_size=32, shuffle=True,
        validation_data=(X_test, X_test), verbose=1
    )
    print(f"Training loss: {training.history['loss'][-1]:.4f}")
    print(f"Validation loss: {training.history['val_loss'][-1]:.4f}")

    n_samples = 1000
    # Compute the reconstruction error of noise data
    samples = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 28, 28, 1))
    reconstruction_error_noise = compute_reconstruction_error(samples, autoencoder)

    # Save and print the autoencoder metrics
    reconstruction_error = compute_reconstruction_error(X_test, autoencoder)
    autoencoder_metrics = {
        "reconstruction_error_noise": format_metric(reconstruction_error_noise),
        "reconstruction_error": format_metric(reconstruction_error),
    }

    pprint(autoencoder_metrics)
    autoencoder.save(os.path.join(weights_path_to_save, 'autoencoder.h5'))
    return autoencoder 


def train_countergan(model, 
                     classifier, 
                     discriminator, 
                     generator, 
                     autoencoder, 
                     batches,
                     test_set,
                     num_classes,
                     target_class,   
                     weighted_version=False,
                     n_iterations=100, 
                     check_every_n_iter=10, 
                     n_discriminator_steps=2,
                     n_generator_steps=3,
                     negative_batches=None, positive_batches=None,
                    ):
    """ 
    Functio for the training of the CounteRGAN
    """

    def check_divergence(x_generated):
        return np.all(np.isnan(x_generated))

    def print_training_information(generator, classifier, test_set, iteration):
        if len(test_set.shape) != 4:
            test_set = np.expand_dims(test_set, axis=-1).astype(np.float16)
        print(test_set.dtype)
        X_gen = generator.predict(test_set)
        clf_pred_test = classifier.predict(test_set) # y_M
        clf_pred = classifier.predict(X_gen)  # y_cf

        delta_clf_pred = (clf_pred - clf_pred_test)[:, target_class]
        #y_target = to_categorical([target_class] * len(clf_pred), num_classes=num_classes)
        print('='*88)

        # Plot original images, residuals, and generated images
        sample_indices  = np.random.choice(len(test_set), 10, replace=False)
        residuals = X_gen[sample_indices]-test_set[sample_indices]

        plot_generated_images(test_set[sample_indices], n_plots=10, n_plots_per_row=10)
        plot_generated_images(residuals, n_plots=10, n_plots_per_row=10, residuals=True)
        plot_generated_images(X_gen[sample_indices], n_plots=10, n_plots_per_row=10)

        reconstruction_error = np.mean(compute_reconstruction_error(X_gen, autoencoder))
        print(f"Autoencoder reconstruction error (infinity to 0): {reconstruction_error:.3f}")
        print(f"Counterfactual prediction gain (0 to 1): {delta_clf_pred.mean():.3f}")
        print(f"Sparsity (L1, infinity to 0): {np.mean(np.abs(X_gen-test_set)):.3f}")

    for iteration in range(n_iterations):
        if iteration > 0:
            x_generated = generator.predict(x_fake_input)
            if check_divergence(x_generated):
                print("Training diverged with the following loss functions:")
                # print(discrim_loss_1, discrim_accuracy, gan_loss,
                #     discrim_loss, discrim_loss_2, clf_loss)
                break

        # Periodically print and plot training information
        if (iteration % check_every_n_iter == 0) or (iteration == n_iterations - 1):
            print_training_information(generator, classifier, test_set, iteration)

        # Train the discriminator

        discriminator.trainable = True
        for _ in range(n_discriminator_steps):
            # if weighted_version:
            #     x_fake_input, _ = next(negative_batches)
            #     x_real, _ = next(positive_batches)
            # else:
            x_fake_input, _ = next(batches)
            x_real = x_fake_input

            x_fake = generator(x_fake_input)

            x_batch = np.concatenate([x_real, x_fake])
            y_batch = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_fake))])

            # Shuffle real and fake examples
            p = np.random.permutation(len(y_batch))
            x_batch, y_batch = x_batch[p], y_batch[p]
            discriminator.train_on_batch(x_batch, y_batch)

        # Train the generator
        discriminator.trainable = False
        for _ in range(n_generator_steps):
            x_fake_input, _ = next(batches)
            y_fake = np.ones(len(x_fake_input))
            y_target = to_categorical([target_class] * len(x_fake_input), num_classes=num_classes)
            model.train_on_batch(x_fake_input, [y_fake, y_target])

    return model
