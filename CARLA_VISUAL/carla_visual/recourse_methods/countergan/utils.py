import numpy as np

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

