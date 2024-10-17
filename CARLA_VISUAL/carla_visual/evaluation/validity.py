import numpy as np
import pandas as pd

import torch 
from torch import nn


def calculate_validity(predicted_labels: torch.Tensor, 
                       target_labels: torch.Tensor):
    """
    Calculates the fraction of counterfactual images that correctly flip the classier's desicion 
    """
    try:
        predicted_labels.shape == target_labels.shape
    except:
        print('Shape mismatch between the  and target_labels') 

    hammimg_distance = torch.sum(target_labels != predicted_labels, axis=1)
    # if hamming distance is 0, two binary vectors are identical
    correct_predictions_num = torch.where(hammimg_distance==0)[0].shape[0]
    validity_rate = correct_predictions_num / predicted_labels.shape[0]

    return round(validity_rate, 3)

