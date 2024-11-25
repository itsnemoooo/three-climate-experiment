import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DNN_3, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim, 512))
        self.layer5 = nn.Sequential(nn.Linear(512, 512))
        self.layer6 = nn.Sequential(nn.Linear(512, action_dim))
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)        
        return x