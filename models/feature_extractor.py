import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(
                in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
        )

        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 128, dropout_rate=dropout_rate),
            nn.ReLU(),
            nn.Linear(128, 128, dropout_rate=dropout_rate),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.max_pool(x)
        x = x.squeeze(-1)

        x = self.linear_layers(x)
        x_norm = F.normalize(x, p=2, dim=1)
        return x_norm
