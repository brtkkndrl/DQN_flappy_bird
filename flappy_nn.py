#
#   PyTorch CNN model.
#

import torch
import torch.nn as nn

class FlappyBirdCNN(nn.Module):
    """
        CNN for extracting feature vector, architecture based on CNN to play Atari Games.
    """
    def __init__(self, observation_shape, features_dim=256, filters=[16, 32]):
        super(FlappyBirdCNN, self).__init__()

        self.features_dim = features_dim

        in_channels, _, _ = observation_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # pass to calculate cnn output size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_shape)
            n_flatten = self.cnn(sample).shape[1]

        # final pass to features dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))