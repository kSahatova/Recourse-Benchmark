import torch
from torch import nn


class AE(nn.Module):
	"""
	Implementation from https://github.com/EoinKenny/AAAI-2021/blob/master/local_models.py
	"""
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True)
		)

		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),

			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		x = torch.sigmoid(x)
		return x