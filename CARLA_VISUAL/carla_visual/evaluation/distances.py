import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.covariance import MinCovDet, EmpiricalCovariance

from carla_visual.dataloaders import load_online_dataset


def calculate_proximity(factuals, counterfactuals):
    return sum((factuals - counterfactuals)**2)

def calculate_sparsity(factuals, counterfcatuals):
    factuals = factuals.flatten()
    counterfcatuals = counterfcatuals.flatten()
    
    return torch.linalg.norm(factuals - counterfcatuals, ord=1)


def calculate_covariance(target_class_samples):
    cov_estimator = MinCovDet().fit(target_class_samples)
    inv_cov = cov_estimator.precision_
    return cov_estimator.covariance_, inv_cov


if __name__ == "__main__":
    output_dir = r'D:\PycharmProjects\XAIRobustness\CARLA_VISUAL\carla_visual\evaluation'

    num_classes = 10 
    num_epochs = 30
    ds_name = 'MNIST'
    data_root = 'D:\PycharmProjects\XAIRobustness\data\images'
    batch_size = 256

    train_data, test_data = load_online_dataset(ds_name, data_root, download=False)

    target_class = 1
    test_indices = torch.nonzero(test_data.targets==target_class)
    target_test_images = test_data.data[test_indices].float().squeeze()
    print(target_test_images.flatten().shape)

    covr, inv_covr = calculate_covariance(target_test_images.flatten().reshape(-1, 1))
    print(covr.shape)
