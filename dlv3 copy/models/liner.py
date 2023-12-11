import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, num_classes=3):
        super(Linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10864, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300),
            nn.ReLU(),
            nn.Linear(300, num_classes)
        )

    def forward(self, img):
        # print(img.shape)
        # print(img.view(img.shape[0], -1).shape)
        output = self.fc(img.view(img.shape[0], -1))
        # print(4,output.shape)
        return output