from matplotlib import offsetbox
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import torch
from torch import nn

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import Input

"""
class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Calculate the size of the output from conv layers
        conv_output_size = self._get_conv_output(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
"""
    

class SimpleCNN(models.Sequential):
    def __init__(self, num_classes=10, **kwargs):
        super(SimpleCNN, self).__init__(**kwargs)

        self.add(layers.Conv2D(64, kernel_size=2, padding='same', activation='relu',
                               name='conv1'))
        self.add(layers.MaxPooling2D(pool_size=2, name='maxpool1'))
        self.add(layers.Dropout(0.3, name='drpt1'))

        self.add(layers.Conv2D(32, kernel_size=2, padding='same', activation='relu', name='conv2'))
        self.add(layers.MaxPooling2D(pool_size=2, name='maxpool2'))
        self.add(layers.Dropout(0.3, name='drpt2'))

        self.add(layers.Flatten(name='flatten1'))
        self.add(layers.Dense(256, activation='relu', name='dense1'))
        self.add(layers.Dropout(0.5, name='drpt3'))
        self.add(layers.Dense(num_classes, activation='softmax', name='dense2'))

        self.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
                     


class Autoencoder(models.Sequential):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            Input(input_shape),
            layers.Dense(32, activation="relu"),
            layers.Dense(8)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(input_shape[0], activation="tanh")
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)


if __name__ == "__main__":
    cnn = SimpleCNN(input_shape=(28, 28, 1), num_classes=10)
    autoencoder = Autoencoder(input_shape=(28, 28, 1))
    autoencoder.compile(optimizer=optimizers.Nadam(), loss='mse')

    print(autoencoder.encoder.summary())
        




