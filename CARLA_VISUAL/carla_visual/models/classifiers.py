import torch
from torch import nn

from carla_visual.models.api import AbstractCNN


class PIECECNN(AbstractCNN):
    def __init__(self, input_channels, num_classes):
        super().__init__(input_channels, num_classes)

        self.input_channels = input_channels
        self.num_classes = num_classes

    def build_conv_layers(self):
        # input is Z, going into a convolution
        self.main = self.features(
            nn.Conv2d(self.input_channels, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

    def build_classifier(self):
        self.classifier = self.classifier(
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x):   
            x = self.main(x)
            x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)  # GAP Layer
            logits = self.classifier(x)
            return logits, x
    
    def get_params_num(self):

        features_params = self.main.parameters()
        clf_params = self.classifier.parameters()
        total_params = sum(p.numel() for p in features_params if p.requires_grad) + \
                        sum(p.numel() for p in clf_params if p.requires_grad)
        return total_params
    



