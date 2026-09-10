"""End-to-end final DIFTER model and its exact training/inference paths."""

from collections.abc import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import masked_mean
from .ccif import CCIF
from .cecc import CECCProjectionHead
from .cei import CompositionalEnvironmentIntervention, SourceEnvironmentRegistry, build_intervention_plans
from .classifier import CosineClassifier
from .hptf import HPTFEncoder
from ..losses.hsic import class_conditional_hsic_loss, cross_covariance_penalty
from ..losses.contrastive import cecc_loss
from ..losses.consistency import cei_losses


class DIFTER(nn.Module):
    def __init__(self, num_classes: int, environment_cardinalities: Mapping[str, int],
                 backbone: nn.Module | None = None, max_windows: int = 16,
                 feature_dim: int = 768, stable_dim: int = 256):
        super().__init__()
        self.max_windows = int(max_windows)
        self.backbone = backbone or HPTFEncoder(hidden_size=feature_dim)
        self.factorizer = CCIF(environment_cardinalities, feature_dim=feature_dim, stable_dim=stable_dim)
        self.classifier = CosineClassifier(stable_dim, num_classes)
        self.cecc_projection = CECCProjectionHead(stable_dim, 256, 128)
        self.cei = CompositionalEnvironmentIntervention(self.factorizer, self.classifier)
        self.source_environment_registry = SourceEnvironmentRegistry(environment_cardinalities)

    def register_source_environments(self, tuples):
        self.source_environment_registry = SourceEnvironmentRegistry(self.factorizer.environment_factors, tuples)

    def encode_windows(self, batch):
        shape = batch["token_ids"].shape
        if len(shape) != 3 or shape[1] > self.max_windows:
            raise ValueError("token_ids must have shape [batch, K<=max_windows, length]")
        flat = {name: batch[name].reshape(shape[0] * shape[1], shape[2])
                for name in ("token_ids", "segments", "delta_ts", "pkt_len", "packet_start")}
        output = self.backbone(flat["token_ids"], flat["segments"], flat["delta_ts"], flat["pkt_len"], flat["packet_start"])
        return output.reshape(shape[0], shape[1], -1)

    def encode_flows(self, batch):
        return masked_mean(self.encode_windows(batch), batch["window_mask"])

    def inference(self, batch):
        """Classifier-only prediction path: no environment labels, CECC, or CEI."""
        h_flow = self.encode_flows(batch)
        stable = self.factorizer.stable_projector(h_flow)
        return self.classifier(stable)

    def forward(self, batch):
        return self.inference(batch)

    def support_forward(self, support_batch):
        """Auxiliary support path with the shared backbone explicitly detached."""
        with torch.no_grad():
            h_flow = self.encode_flows(support_batch)
        return self.factorizer(h_flow.detach())

    def training_objective(self, main_batch, main_labels, support_batch, support_labels,
                           support_environment, scales, weights, training_step=0):
        h_main = self.encode_flows(main_batch)
        stable_main = self.factorizer.stable_projector(h_main)
        logits = self.classifier(stable_main)
        margin_logits = logits.clone()
        margin_logits[torch.arange(main_labels.numel(), device=main_labels.device), main_labels] -= self.classifier.scale * float(weights.get("cosine_margin", 0.10))
        classification = F.cross_entropy(margin_logits, main_labels)

        support = self.support_forward(support_batch)
        invariant, environment, orthogonal = [], [], []
        for factor, values in support_environment.items():
            invariant.append(class_conditional_hsic_loss(support["stable"], support_labels, values,
                int(weights.get("min_class_samples", 4)), int(weights.get("max_samples_per_class", 32))))
            environment.append(F.cross_entropy(support["environment_logits"][factor], values))
            orthogonal.append(cross_covariance_penalty(support["stable"], support["environment"][factor]))
        zero = support["stable"].sum() * 0.0
        inv = torch.stack(invariant).mean() if invariant else zero
        env = torch.stack(environment).mean() if environment else zero
        orth = torch.stack(orthogonal).mean() if orthogonal else zero
        rec = F.mse_loss(support["reconstruction"], support["reconstruction_target"])
        ccif = (float(weights["lambda_inv"]) * inv + float(weights["lambda_env"]) * env +
                float(weights["lambda_orth"]) * orth + float(weights["lambda_rec"]) * rec)

        environment_keys = _joint_environment_keys(support_environment)
        cecc = cecc_loss(self.cecc_projection(support["stable"]), support_labels, environment_keys,
                         float(weights["temperature"]))
        plans = build_intervention_plans(support_labels, support_environment,
                                         self.source_environment_registry, training_step)
        cei_output = self.cei(support, plans, support_labels)
        cei_cls, cei_sem, cei_env = cei_losses(cei_output)
        cei = (float(weights["lambda_cei_cls"]) * cei_cls + float(weights["lambda_cei_sem"]) * cei_sem +
               float(weights["lambda_cei_env"]) * cei_env)
        total = classification + scales.ccif * ccif + scales.cecc * float(weights["lambda_cecc"]) * cecc + scales.cei * cei
        return {"loss": total, "classification": classification, "ccif": ccif,
                "cecc": cecc, "cei": cei, "logits": logits, "stable_main": stable_main}


def _joint_environment_keys(factors):
    if not factors:
        raise ValueError("at least one source-environment factor is required")
    names = tuple(factors)
    tuples = list(zip(*[factors[name].detach().cpu().tolist() for name in names]))
    mapping = {value: index for index, value in enumerate(sorted(set(tuples)))}
    first = factors[names[0]]
    return torch.tensor([mapping[value] for value in tuples], device=first.device, dtype=torch.long)
