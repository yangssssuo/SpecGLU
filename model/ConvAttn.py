import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvInput(nn.Module):
    def __init__(self, input_len=4000, d_model=128, kernel_size=9, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [B, L]
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)    # [B, d_model, L']
        x = self.norm(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)  # [B, L', d_model]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=2000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class ConvInputSpectralTransformer(nn.Module):
    def __init__(self, input_len=4000, d_model=128, n_layers=6, n_heads=4, kernel_size=3, stride=2, num_outputs=1, dropout=0.1):
        super().__init__()
        self.conv_proj = ConvInput(input_len, d_model, kernel_size, stride)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len // stride + 10)
        self.encoder = nn.Sequential(*[
            TransformerBlock(d_model, heads=n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, num_outputs)

    def forward(self, x):
        # x: [B, L]
        x = self.conv_proj(x)        # [B, L', D]
        x = self.pos_encoder(x)     # [B, L', D]
        x = self.encoder(x)         # [B, L', D]
        x = x.transpose(1, 2)       # [B, D, L']
        x = self.pool(x).squeeze(-1)  # [B, D]
        return self.output(x)       # [B, num_outputs]
