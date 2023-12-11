import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes=3):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            # nn.Linear(4352, 240),
            # nn.ReLU(),
            # nn.Linear(240, 84),
            # nn.ReLU(),
            nn.Linear(2016, num_classes)
        )

    def forward(self, img):
        # print(1,img.shape)
        img = torch.unsqueeze(img,dim=1)
        feature = self.conv(img)
        print(2,feature.shape)
        print(3,feature.view(img.shape[0], -1).shape)
        output = self.fc(feature.view(img.shape[0], -1))
        # print(4,output.shape)
        return output