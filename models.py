import torch
import torch.nn as nn
import numpy as np

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_features, out_features, routing_iters):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, out_features),
                nn.BatchNorm1d(out_features)
            ) for _ in range(num_capsules)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u_squash = self.squash(u)
        return u_squash

    def squash(self, x, epsilon=1e-7):
        squared_norm = (x ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        vector = scale * x / torch.sqrt(squared_norm + epsilon)
        return vector

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class CapsResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CapsResNet, self).__init__()
        self.conv1 = nn.Sequential(
            ResidualBlock(1, 32, stride=1),
            ResidualBlock(32, 48, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            ResidualBlock(48, 64, stride=1),
            ResidualBlock(64, 192, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            ResidualBlock(192, 256, stride=1),
            ResidualBlock(256, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc_size = self._get_fc_size()  # Ensure this size is correct
        self.capsule_layer = CapsuleLayer(num_capsules=num_classes, num_features=self.fc_size, out_features=10, routing_iters=3)
        self.fc = nn.Linear(10 * 10, 32)
        self.out = nn.Linear(32, num_classes)

    def _get_fc_size(self):
        with torch.no_grad():
            dummy_data = torch.zeros(1, 1, 28, 28)  # Adjust based on input size
            dummy_data = self.conv1(dummy_data)
            dummy_data = self.conv2(dummy_data)
            dummy_data = self.conv3(dummy_data)
            return int(np.prod(dummy_data.size()[1:]))  # Consider only channel and spatial dimensions

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
