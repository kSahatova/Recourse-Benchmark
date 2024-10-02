import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Dict, Optional



def evaluate_model(model: nn.Module, 
                   data_loader: DataLoader, 
                   device: str,
                   criterion: Optional[nn.Module] = None) -> Dict[str, float]:
    """
    Evaluate a PyTorch model for accuracy and optionally loss.

    Args:
    model (nn.Module): The PyTorch model to evaluate.
    data_loader (DataLoader): The DataLoader containing the evaluation data.
    device (str): The device to run the evaluation on ('cuda' or 'cpu').
    criterion (nn.Module, optional): Loss function. If provided, loss will be calculated.

    Returns:
    Dict[str, float]: A dictionary containing 'accuracy' and optionally 'loss'.
    """
    model.eval()  # Set the model to evaluation mode

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs[0], 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

    # Calculate final metrics
    accuracy = total_correct / total_samples
    results = {"accuracy": accuracy}

    if criterion is not None:
        avg_loss = total_loss / total_samples
        results["loss"] = avg_loss

    return results