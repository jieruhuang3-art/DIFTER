from collections.abc import Mapping, Sequence
import torch
import torch.nn as nn


class StableProjector(nn.Sequential):
    def __init__(self, feature_dim=768, hidden_dim=512, stable_dim=256, dropout=0.1):
        super().__init__(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden_dim, stable_dim), nn.LayerNorm(stable_dim))


class EnvironmentProjector(nn.Sequential):
    def __init__(self, feature_dim=768, hidden_dim=256, environment_dim=128):
        super().__init__(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
                         nn.Linear(hidden_dim, environment_dim))


class EnvironmentAwareReconstructor(nn.Module):
    def __init__(self, factors: Sequence[str], feature_dim=768, stable_dim=256,
                 environment_dim=128, hidden_dim=512):
        super().__init__()
        self.factors = tuple(factors)
        self.neutral = nn.ParameterDict({factor: nn.Parameter(torch.zeros(environment_dim)) for factor in self.factors})
        self.decoder = nn.Sequential(nn.Linear(stable_dim + len(self.factors) * environment_dim, hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, feature_dim))

    def forward(self, stable, environment: Mapping[str, torch.Tensor], valid_masks=None):
        pieces, valid_masks = [stable], valid_masks or {}
        for factor in self.factors:
            value, mask = environment[factor], valid_masks.get(factor)
            if mask is None:
                pieces.append(value)
            else:
                mask = mask.to(value.device, torch.bool).view(-1, 1)
                neutral = self.neutral[factor].to(value.dtype).view(1, -1).expand_as(value)
                pieces.append(torch.where(mask, value, neutral))
        return self.decoder(torch.cat(pieces, -1))


class CCIF(nn.Module):
    """Class-Conditional Invariant Factorization."""

    def __init__(self, environment_cardinalities: Mapping[str, int], feature_dim=768,
                 stable_hidden=512, stable_dim=256, env_hidden=256, env_dim=128, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.environment_factors = tuple(environment_cardinalities)
        self.stable_projector = StableProjector(feature_dim, stable_hidden, stable_dim, dropout)
        self.environment_projectors = nn.ModuleDict({
            factor: EnvironmentProjector(feature_dim, env_hidden, env_dim) for factor in self.environment_factors
        })
        self.environment_heads = nn.ModuleDict({
            factor: nn.Linear(env_dim, count) for factor, count in environment_cardinalities.items() if count >= 2
        })
        self.reconstructor = EnvironmentAwareReconstructor(self.environment_factors, feature_dim, stable_dim, env_dim, stable_hidden)

    def factorize(self, h_flow):
        stable = self.stable_projector(h_flow)
        environment = {factor: projector(h_flow) for factor, projector in self.environment_projectors.items()}
        logits = {factor: self.environment_heads[factor](value) for factor, value in environment.items()
                  if factor in self.environment_heads}
        return {"stable": stable, "environment": environment, "environment_logits": logits}

    def reconstruct(self, stable, environment, valid_masks=None):
        return self.reconstructor(stable, environment, valid_masks)

    def forward(self, h_flow, factor_valid_masks=None):
        output = self.factorize(h_flow)
        return {**output, "reconstruction": self.reconstruct(output["stable"], output["environment"], factor_valid_masks),
                "reconstruction_target": h_flow.detach()}
