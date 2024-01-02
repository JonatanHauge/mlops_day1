from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
    
