import torch
import torch.nn as nn
import numpy as np
from models.residual_block import ResidualBlock
from models.capsule_layer import CapsuleLayer

class CapsResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dim=32):
        super(CapsResNet, self).__init__()
        self.conv1 = nn.Sequential(
            ResidualBlock(input_channels, 32),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            ResidualBlock(64, 256),
            ResidualBlock(256, 192),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            ResidualBlock(192, 256),
            ResidualBlock(256, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.fc_size = self._get_fc_size(input_channels, dim)
        self.capsule_layer = CapsuleLayer(num_capsules=num_classes, num_features=self.fc_size, out_features=10)
        self.fc = nn.Linear(10 * 10, 32)
        self.out = nn.Linear(32, num_classes)

    def _get_fc_size(self, input_channels, dim):
        with torch.no_grad():
            dummy_data = torch.zeros(1, input_channels, dim, dim)
            dummy_data = self.conv1(dummy_data)
            dummy_data = self.conv2(dummy_data)
            dummy_data = self.conv3(dummy_data)
            return int(np.prod(dummy_data.size()[1:]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.capsule_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out(x)
        return x
