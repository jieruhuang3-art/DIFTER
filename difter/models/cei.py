"""Parameter-free compositional environment intervention."""

from dataclasses import dataclass
from collections.abc import Mapping
import itertools
import hashlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InterventionPlan:
    anchor_index: int
    donor_indices: Mapping[str, int]
    environment_targets: Mapping[str, int]


class SourceEnvironmentRegistry:
    def __init__(self, factors, observed=()):
        self.factors = tuple(factors)
        self.observed = {tuple(value) for value in observed}

    def is_global_source_unobserved(self, value):
        return tuple(value) not in self.observed


def build_intervention_plans(labels, factor_values, source_registry=None, training_step=0, seed=42):
    """Build the source-only compositional plans used by CEI."""
    factors = tuple(factor_values)
    registry = source_registry or SourceEnvironmentRegistry(factors)
    digest = hashlib.sha256(f"{seed}\0{training_step // 136}\0{training_step}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    cpu_values = {factor: value.detach().cpu().tolist() for factor, value in factor_values.items()}
    cpu_labels = labels.detach().cpu().tolist()
    plans = []
    for anchor in range(labels.numel()):
        peers = [index for index in range(labels.numel()) if index != anchor and cpu_labels[index] == cpu_labels[anchor]]
        original = tuple(cpu_values[factor][anchor] for factor in factors)
        selected = None
        if len(factors) >= 2 and peers:
            available = [sorted({cpu_values[factor][anchor], *[cpu_values[factor][index] for index in peers]}) for factor in factors]
            candidates = list(itertools.product(*available)); generator.shuffle(candidates)
            candidates = [value for value in candidates if value != original and registry.is_global_source_unobserved(value)]
            candidates.sort(key=lambda value: sum(a != b for a, b in zip(value, original)), reverse=True)
            for candidate in candidates:
                donors = {}
                for position, factor in enumerate(factors):
                    if candidate[position] == original[position]: continue
                    choices = [index for index in peers if cpu_values[factor][index] == candidate[position]]
                    if not choices: donors = {}; break
                    donors[factor] = generator.choice(choices)
                if donors: selected = (candidate, donors); break
        if selected is None:
            order = list(factors); generator.shuffle(order)
            for factor in order:
                choices = [index for index in peers if cpu_values[factor][index] >= 0 and cpu_values[factor][index] != cpu_values[factor][anchor]]
                if choices:
                    donor = generator.choice(choices); candidate = list(original)
                    candidate[factors.index(factor)] = cpu_values[factor][donor]
                    selected = (tuple(candidate), {factor: donor}); break
        if selected is not None:
            candidate, donors = selected
            targets = {factor: int(candidate[factors.index(factor)]) for factor in donors}
            plans.append(InterventionPlan(anchor, donors, targets))
    return tuple(plans)


class CompositionalEnvironmentIntervention(nn.Module):
    def __init__(self, factorizer, classifier):
        super().__init__()
        object.__setattr__(self, "factorizer", factorizer)
        object.__setattr__(self, "classifier", classifier)

    @staticmethod
    def _detached_cosine(value, classifier):
        weight = F.normalize(classifier.weight.detach(), dim=-1) * classifier.scale.detach()
        return F.linear(F.normalize(value, dim=-1), weight)

    def forward(self, factorized, plans, class_labels):
        if not plans:
            zero = factorized["stable"][:0]
            return {"valid": False, "zero_reference": factorized["stable"], "stable_cf": zero}
        anchors = torch.tensor([plan.anchor_index for plan in plans], device=class_labels.device)
        stable_anchor = factorized["stable"].index_select(0, anchors)
        intervened, swapped_masks, targets = {}, {}, {}
        for factor, values in factorized["environment"].items():
            base = values.index_select(0, anchors)
            rows, swapped, factor_targets = [], [], []
            for row, plan in enumerate(plans):
                donor = plan.donor_indices.get(factor)
                if donor is None:
                    rows.append(base[row]); swapped.append(False); factor_targets.append(-1)
                else:
                    rows.append(values[donor].detach()); swapped.append(True)
                    factor_targets.append(plan.environment_targets[factor])
            intervened[factor] = torch.stack(rows)
            swapped_masks[factor] = torch.tensor(swapped, device=values.device, dtype=torch.bool)
            targets[factor] = torch.tensor(factor_targets, device=values.device)
        h_cf = self.factorizer.reconstruct(stable_anchor, intervened)
        refactorized = self.factorizer.factorize(h_cf)
        environment_logits = {}
        for factor, head in self.factorizer.environment_heads.items():
            value = refactorized["environment"][factor]
            environment_logits[factor] = F.linear(value, head.weight.detach(), None if head.bias is None else head.bias.detach())
        return {"valid": True, "zero_reference": factorized["stable"], "h_cf": h_cf,
                "stable_cf": refactorized["stable"], "stable_target": stable_anchor.detach(),
                "class_labels": class_labels.index_select(0, anchors),
                "classifier_logits": self._detached_cosine(refactorized["stable"], object.__getattribute__(self, "classifier")),
                "environment_logits": environment_logits, "environment_targets": targets,
                "swapped_masks": swapped_masks}
