import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, learnable_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        initial = torch.tensor(math.log(10.0))
        if learnable_scale:
            self.log_s = nn.Parameter(initial)
        else:
            self.register_buffer("log_s", initial)

    @property
    def scale(self):
        return self.log_s.exp()

    def clamp_scale_(self):
        with torch.no_grad():
            self.log_s.clamp_(math.log(1.0), math.log(30.0))

    def forward(self, value):
        return self.scale * F.linear(F.normalize(value, dim=-1), F.normalize(self.weight, dim=-1))
