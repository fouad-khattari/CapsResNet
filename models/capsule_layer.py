import torch
import torch.nn as nn

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_features, out_features, routing_iters=3):
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
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        return self.squash(u)

    def squash(self, x, epsilon=1e-7):
        squared_norm = (x ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + epsilon)
