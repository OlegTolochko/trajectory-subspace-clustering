import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # reduces spatial dimension F by 6, padding=1 could be thus considered, so no spatial reduction is performed
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        x = self.max_pool(x)
        x = x.squeeze(-1)
        
        x = self.linear_layers(x)
        return x
