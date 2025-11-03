import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockWithAttention(nn.Module):
    def __init__(self, dim, dropout=0.1, num_heads=4):
        super().__init__()

        # 原始 MLP 路径
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # 自注意力路径
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, feature_dim)

        # ---- 1. MLP 路径 ----
        residual = x
        x_mlp = F.relu(self.norm1(self.linear1(x)))
        x_mlp = self.dropout(x_mlp)
        x_mlp = self.norm2(self.linear2(x_mlp))

        # ---- 2. Attention 路径 ----
        # reshape to sequence: (batch, seq_len, dim)
        x_seq = x.unsqueeze(1)  # (B, 1, D)
        x_norm = self.attn_norm(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # Self-attention
        attn_out = attn_out.squeeze(1)  # (B, D)

        # ---- 3. 路径融合 ----
        out = x_mlp + attn_out + residual
        return F.relu(out)



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim*2)
        self.norm1 = nn.LayerNorm(dim*2)
        self.linear2 = nn.Linear(dim*2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.celu(self.norm1(self.linear1(x)))
        x = self.dropout(x)
        x = self.norm2(self.linear2(x))
        return F.celu(x + residual)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=4000, hidden_dim=2048, num_blocks=10, num_outputs=1, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        x = F.celu(self.input_layer(x))
        x = self.blocks(x)
        out = self.output_layer(x)
        return out
