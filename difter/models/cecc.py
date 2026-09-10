import torch.nn as nn
import torch.nn.functional as F


class CECCProjectionHead(nn.Module):
    def __init__(self, stable_dim=256, hidden_dim=256, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(stable_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim))

    def forward(self, stable):
        return F.normalize(self.projection(stable), dim=-1)
