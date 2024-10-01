import os
import json

import numpy as np
from typing import Union
from pprint import pprint

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


def compute_reconstruction_error(x, autoencoder):
    """
    Compute the reconstruction error for a given autoencoder and data points.
    """
    preds = autoencoder.predict(x)
    preds_flat = preds.reshape((preds.shape[0], -1))
    x_flat = x.reshape((x.shape[0], -1))
    return np.linalg.norm(x_flat - preds_flat, axis=1)


def format_metric(metric):
    """
    Return a formatted version of a metric, with the confidence interval.
    """
    return f"{metric.mean():.3f} ± {1.96*metric.std()/np.sqrt(len(metric)):.3f}"


def compute_metrics(samples, counterfactuals, desired_class, latencies,
                    classifier, autoencoder, batch_latency=None):
    """ Summarize the relevant metrics in a dictionary. """
    reconstruction_error = compute_reconstruction_error(counterfactuals, autoencoder)
    delta = np.abs(samples-counterfactuals)
    l1_distances = delta.reshape(delta.shape[0], -1).sum(axis=1)
    prediction_gain = (
        classifier.predict(counterfactuals)[:, desired_class] -
        classifier.predict(samples)[:, desired_class]
    )

    metrics = dict()
    metrics["reconstruction_error"] = format_metric(reconstruction_error)
    metrics["prediction_gain"] = format_metric(prediction_gain)
    metrics["sparsity"] = format_metric(l1_distances)
    metrics["latency"] = format_metric(latencies)
    batch_latency = batch_latency if batch_latency else sum(latencies)
    metrics["latency_batch"] = f"{batch_latency:.3f}"

    return metrics


def save_experiment(classifier, autoencoder, experiment_path,
                    samples, counterfactuals, latencies):
    """Create an experiment folder and save counterfactuals, latencies and metrics."""
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    np.save(os.path.join(experiment_path, 'counterfactuals.npy'), counterfactuals)
    np.save(os.path.join(experiment_path, 'latencies.npy'), latencies)

    metrics = compute_metrics(samples, counterfactuals, latencies, classifier, autoencoder)
    json.dump(metrics, open(os.path.join(experiment_path, 'metrics.json'), "w"))
    pprint(metrics)


def load_autoencoder(weights_path: str) -> Union[Model, Sequential, None]:
    """
    Loads pretrained model
    :param weights_path: specifies path to the pretrained weights
    :return: successful loading returns Model, None - otherwise
    """
    # Load the model
    try:
        loaded_model = load_model(weights_path)
        print(loaded_model.summary())
        return loaded_model
    except FileNotFoundError as e:
        print(e)
    except ImportError as e:
        print(f"ImportError: {e}")
