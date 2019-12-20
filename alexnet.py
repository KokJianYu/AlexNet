import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        # Alexnet contains of 8 layers. 5 Conv, and 3 FC
        self.conv1 = nn.Conv2d(1,1,1) # (in_channels, out_channels, kernel_size) https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        self.conv2 = nn.Conv2d(1,1,1)
        self.conv3 = nn.Conv2d(1,1,1)
        self.conv4 = nn.Conv2d(1,1,1)
        self.conv5 = nn.Conv2d(1,1,1)
        self.fc1 = nn.Linear(1,1) # (in_features, out_features) # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        self.fc2 = nn.Linear(1,1)
        self.fc3 = nn.Linear(1,1)

    def forward(self, x):
        # Do forward pass here. 
        pass
    