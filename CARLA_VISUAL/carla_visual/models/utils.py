import os
import requests

import torch
from torch import nn


def fetch_weights(url: str, file_name:str, output_dir: str):
	data = requests.get(url)
	with open(os.path.join(output_dir, file_name), 'wb') as file:
		file.write(data.content)
		


def load_model_weights(model: nn.Module, weights_path: str, device: str) -> nn.Module:
	model.load_state_dict(torch.load(weights_path, map_location=device))
	return model







