import torch
import torch.nn.functional as F


class PrototypeEMA:
    """Detached normalized source-train class prototypes."""

    def __init__(self, classes, dimension, device, mu=0.95):
        self.mu = float(mu)
        self.prototypes = torch.zeros(classes, dimension, device=device)
        self.initialized = torch.zeros(classes, dtype=torch.bool, device=device)
        self.sample_count = torch.zeros(classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, features, labels):
        features = F.normalize(features.detach().float(), dim=-1)
        for class_id in labels.unique().tolist():
            current = features[labels == class_id]
            mean = F.normalize(current.mean(0), dim=0)
            if self.initialized[class_id]:
                mean = F.normalize(self.mu * self.prototypes[class_id] + (1 - self.mu) * mean, dim=0)
            self.prototypes[class_id] = mean; self.initialized[class_id] = True
            self.sample_count[class_id] += current.size(0)

    def feature_loss(self, features, labels):
        valid = self.initialized[labels]
        if not valid.any(): return features.sum() * 0.0
        return (1 - (F.normalize(features[valid], dim=-1) * self.prototypes[labels[valid]].detach()).sum(-1)).mean()

    def weight_loss(self, classifier):
        valid = self.initialized & self.sample_count.gt(0)
        if not valid.any(): return classifier.weight.sum() * 0.0
        return (1 - (F.normalize(classifier.weight[valid], dim=-1) * self.prototypes[valid].detach()).sum(-1)).mean()
