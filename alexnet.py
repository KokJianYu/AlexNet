import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add bias

class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super(AlexNet, self).__init__()

        # Alexnet contains of 8 layers. 5 Conv, and 3 FC
        self.conv1 = nn.Conv2d(3,96,11, stride=4) # (in_channels, out_channels, kernel_size) https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        self.conv2 = nn.Conv2d(96,256,5, padding=2)
        self.conv3 = nn.Conv2d(256,384,3,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6,4096) # (in_features, out_features) # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,output_dim)
        self.localResponseNorm = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2)
        self.maxPool = nn.MaxPool2d(3, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        self.setup_bias_and_weights()
    
    def setup_bias_and_weights(self):
        # Set all layer's weight to be normal distribution with mean=0, std=0.01
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)

        # Set the following bias to be 1 as stated in the paper
        nn.init.ones_(self.conv2.bias)
        nn.init.ones_(self.conv4.bias)
        nn.init.ones_(self.conv5.bias)
        nn.init.ones_(self.fc1.bias)
        nn.init.ones_(self.fc2.bias)
        nn.init.ones_(self.fc3.bias)

        # Remaining layers bias to be 0
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv3.bias)


    def forward(self, x):
        # Do forward pass here. 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.localResponseNorm(x)
        x = self.maxPool(x)


        x = self.conv2(x)
        x = F.relu(x)
        x = self.localResponseNorm(x)
        x = self.maxPool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxPool(x)

        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    