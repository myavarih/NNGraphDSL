import torch
import torch.nn as nn


class SplitNet(nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()
        self.left = nn.Linear(in_features=4, out_features=4)
        self.right = nn.Linear(in_features=4, out_features=4)

    def forward(self, x):
        split_0, split_1 = torch.chunk(x, 2, dim=1)
        split_0 = self.left(split_0)
        split_1 = self.right(split_1)
        x = torch.cat([split_0, split_1], dim=1)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = SplitNet().to(device)
    x = torch.randn(1, 1, 8).to(device)
    print(model(x).shape)
