import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # reduces spatial dimension F by 6, padding=1 could be thus considered, so no spatial reduction is performed
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3, stride=1)
        
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 128)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        print(x.shape)
        
        x = self.max_pool(x)
        x = x.squeeze(-1)
        print(x.shape)
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return x
