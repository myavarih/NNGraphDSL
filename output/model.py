import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(normalized_shape=128)
        self.ff1 = nn.Linear(in_features=128, out_features=512)
        self.gelu1 = nn.GELU()
        self.drop2 = nn.Dropout(p=0.1)
        self.ff2 = nn.Linear(in_features=512, out_features=128)
        self.norm2 = nn.LayerNorm(normalized_shape=128)
        self.out = nn.Flatten(start_dim=0, end_dim=1)

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        attn = self.drop1(attn)
        norm1 = self.norm1(x + attn)  # residual_1
        ff1 = self.ff1(norm1)
        ff1 = self.gelu1(ff1)
        ff1 = self.drop2(ff1)
        ff1 = self.ff2(ff1)
        x = self.norm2(norm1 + ff1)  # residual_2
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    model = TransformerEncoder().to(device)
    x = torch.randn(1, 512, 128).to(device)
    print(model(x).shape)
