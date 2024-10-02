import torch 
from torch import nn

from abc import ABC, abstractmethod



class AbstractCNN(nn.Module, ABC):
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential
        self.classifier = nn.Sequential

        self.build_conv_layers()
        self.build_classifier()

    
    @abstractmethod
    def build_conv_layers(self):
        raise NotImplementedError
    
    @abstractmethod
    def build_classifier(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_params_num(self):
        raise NotImplementedError
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
