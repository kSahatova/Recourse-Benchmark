import torch
from torch import nn


class PIECEGenerator(nn.Module):
	"""
	Implementation from https://github.com/EoinKenny/AAAI-2021/blob/master/local_models.py
	"""
	def __init__(self, ngpu, nc=1, nz=100, ngf=64):
		super(PIECEGenerator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
			nn.Tanh()
		)

	def forward(self, input):
		output = self.main(input)
		return output