import torch
import torch.nn as nn
import torch.nn.functional as F

class StabilizationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)

        self.conv4 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 32, 3, padding=1)

        self.out = nn.Conv3d(32, 2, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.out(x)
def motion_loss(pred, flow):
    return torch.mean(torch.abs(pred - flow))

def temporal_smoothness(pred):
    return torch.mean(torch.abs(pred[:, :, 1:] - pred[:, :, :-1]))

def spatial_smoothness(pred):
    dx = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def total_loss(pred, flow, w1=1.0, w2=0.1, w3=0.1):
    return (
        w1 * motion_loss(pred, flow) +
        w2 * temporal_smoothness(pred) +
        w3 * spatial_smoothness(pred)
    )
