import torch.nn.functional as F


def reconstruction_loss(reconstruction, detached_target):
    return F.mse_loss(reconstruction, detached_target)
