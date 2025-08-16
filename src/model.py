import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeFaceCNN(nn.Module):
    def __init__(self):
        super(FakeFaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # we'll determine this dynamically
        self.flatten_dim = None  
        self.fc1 = None
        self.fc2 = None

    def _init_fc(self, x):
        """Initialize FC layers dynamically based on input shape."""
        self.flatten_dim = x.view(x.size(0), -1).shape[1]
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        if self.fc1 is None:
            self._init_fc(x)  # first forward pass determines FC sizes
            self.fc1 = self.fc1.to(x.device)
            self.fc2 = self.fc2.to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
