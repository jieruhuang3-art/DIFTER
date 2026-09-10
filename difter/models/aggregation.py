"""Flow-level aggregation used by the final model."""

import torch


def masked_mean(window_features: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid traffic-window units.

    Args:
        window_features: ``[batch, K, feature_dim]``.
        window_mask: ``[batch, K]`` with one for real windows and zero for padding.
    """
    if window_features.ndim != 3 or window_mask.shape != window_features.shape[:2]:
        raise ValueError("window features/mask shape mismatch")
    mask = window_mask.to(device=window_features.device, dtype=window_features.dtype)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("each flow must contain at least one real window")
    return (window_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
