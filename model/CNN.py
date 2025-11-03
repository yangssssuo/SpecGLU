import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumCNN(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(SpectrumCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=2),
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Conv1d(32, 64, kernel_size=10, stride=2),
            nn.BatchNorm1d(64),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob)  # 添加 Dropout
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 989, 512),
            nn.BatchNorm1d(512),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 Dropout
            nn.Linear(128, 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)  # 扩展维度，将输入变为 (batch_size, 1, 3000)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平为全连接层的输入
        x = self.fc_layers(x)
        return x.view(-1, 1)  # reshape为1x1矩阵
