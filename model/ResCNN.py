import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.dropout = nn.Dropout(0.2)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.dropout(out)
        out += identity
        return self.relu(out)

class DeepIRConvNet(nn.Module):
    def __init__(self, input_channels=1, input_len=4000, hidden_dim=1024):
        super(DeepIRConvNet, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_channels, 32),
            ResidualBlock1D(32, 64),
            # ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128),
            # ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256),
            ResidualBlock1D(256, 512),
            nn.AdaptiveAvgPool1d(1),  # 输出变为 (B, 256, 1)
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),             # (B, 256)
            nn.Linear(512, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # 回归任务
        )

    def forward(self, x):  # 输入: (B, 1, 4000)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.mlp(x)
        return x
