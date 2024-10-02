import torch
from torchvision import transforms 

import os.path as osp
import numpy as np

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from PIL import Image


def plot_image(generated_batch):
    if isinstance(generated_batch, torch.Tensor):
        array = generated_batch.detach().cpu().numpy()

    # Squeeze out any singular dimensions
    array = np.squeeze(array)
    
    # Normalize the array to [0, 255] range
    if array.max() <= 1.0:
        array = (array * 255)
    array = array.astype(np.uint8)
    image = Image.fromarray(array, mode='L')
    plt.imshow(image, cmap='gray')        
    plt.axis('off')
    plt.show()


def plot_misclassifications(misclassified: List[Tuple[torch.Tensor, int, int]],
                                 class_names: Optional[List[str]] = None,
                                 denormalize: Optional[transforms.Normalize] = None):
    """
    Visualize misclassified samples.

    Args:
    misclassified (List[Tuple[torch.Tensor, int, int]]): List of misclassified samples.
    class_names (Optional[List[str]]): List of class names for labels.
    denormalize (Optional[transforms.Normalize]): Denormalization transform, if applicable.
    """
    num_samples = len(misclassified)
    fig, axes = plt.subplots(2, 5, figsize=(10, 8)) if num_samples > 5 else plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    axes = axes.flatten() if num_samples > 5 else axes

    for i, (img, true_label, pred_label) in enumerate(misclassified):
        img = img.cpu().squeeze()
        
        if denormalize:
            img = denormalize(img)

        img = img.numpy()
        #img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray')
        true_class = class_names[true_label] if class_names else f"Class {true_label}"
        pred_class = class_names[pred_label] if class_names else f"Class {pred_label}"
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()