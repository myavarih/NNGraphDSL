import torch
import torch.nn as nn


class QuantMLP(nn.Module):
    def __init__(self):
        super(QuantMLP, self).__init__()
        self.fc1 = torch.quantization.QuantWrapper(nn.Linear(in_features=784, out_features=128))
        self.relu1 = nn.ReLU()
        self.fc2 = torch.quantization.QuantWrapper(nn.Linear(in_features=128, out_features=10))
        self.out = nn.Softmax(dim=1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = self.dequant(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = QuantMLP().to(device)
    x = torch.randn(32, 784).to(device)
    print(model(x).shape)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(2):
        x = torch.randn(32, 784).to(device)
        y = model(x)
        target = torch.zeros_like(y)
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/2, Loss: {loss.item():.4f}')
