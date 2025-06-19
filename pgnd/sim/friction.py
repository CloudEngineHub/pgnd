import torch
import torch.nn as nn


class Friction(nn.Module):
    def __init__(self, mu_init=0.5):
        super(Friction, self).__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu_init))
    
    def clip(self):
        self.mu.data = torch.clamp(self.mu.data, 0, 1)
