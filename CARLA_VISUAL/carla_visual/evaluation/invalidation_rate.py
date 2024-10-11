import numpy as np
from tqdm import tqdm
import tensorflow_probability as tfp
import tensorflow as tf 


def perturb_sample(x, n_samples, sigma2):

    # stack copies of this sample, i.e. n rows of x.
    n = x.shape[1]
    X = np.tile(x, reps=(n_samples, 1, 1, 1))

    # sample normal distributed values
    Sigma = tf.eye(n) * sigma2
    eps = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(n),
            scale_diag=Sigma
                  ).sample((n_samples,))
    eps = tf.expand_dims(eps, axis=-1)
    return X + eps, Sigma, eps


def calculate_invalidation_rate(counterfactuals, classifier):
    sigmas2 = [0.01, 0.015, 0.02, 0.025]
    n_samples = 1000

    rir_results = {}

    for sigma in sigmas2:
        result = []
        for x in tqdm(counterfactuals[:1000]):
            if len(x.shape) < 4:
                x = tf.expand_dims(x, axis=0)
            cf_prediction = (classifier.predict(x, verbose=0) > 0.5).astype(int)
            cf_prediction = np.tile(cf_prediction, (n_samples, 1)) 

            X_pert, _, _ = perturb_sample(x, n_samples, sigma2=sigmas2[0])
            cf_pert_prediction = (classifier.predict(X_pert, verbose=0).squeeze() > 0.5).astype(int)
            delta_M = np.mean(np.all(cf_prediction - cf_pert_prediction, axis=1))

            result.append(delta_M)
        rir_results['sigma_'+str(sigma)] = result
