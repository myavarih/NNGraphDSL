import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.out = nn.Flatten()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)
        conv1 = self.conv2(conv1)
        conv1 = self.bn2(conv1)
        x = conv1 + x
        x = self.relu2(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = ResBlock().to(device)
    x = torch.randn(1, 64, 32, 32).to(device)
    print(model(x).shape)
