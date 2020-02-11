# Author: Daiwei (David) Lu
# A custom network in VGG style

import torch
from torch import nn


class ConvPool(nn.Module):
    def __init(self, channelin, channelout):
        super(Net, self).__init__()
        self.convpool = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.convpool(x)


class Conv(nn.Module):
    def __init(self, channelin, channelout):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 64),
                                        nn.Sigmoid(),
                                        nn.Dropout(0.25),
                                        nn.Linear(64, 7))

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        x1 = self.features(x)
        x2 = self.avgpool(x1)
        x3 = torch.flatten(x2, 1)
        x4 = self.classifier(x3)
        return x4
