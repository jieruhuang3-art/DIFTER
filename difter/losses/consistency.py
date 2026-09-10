import torch
import torch.nn.functional as F


def cei_losses(output):
    zero = output["zero_reference"].sum() * 0.0
    if not output.get("valid", False):
        return zero, zero, zero
    classification = F.cross_entropy(output["classifier_logits"].float(), output["class_labels"])
    semantic = (1.0 - F.cosine_similarity(output["stable_cf"].float(), output["stable_target"].float(), dim=-1)).mean()
    environment_losses = []
    for factor, logits in output["environment_logits"].items():
        mask, target = output["swapped_masks"][factor], output["environment_targets"][factor]
        if mask.any():
            environment_losses.append(F.cross_entropy(logits[mask].float(), target[mask]))
    environment = torch.stack(environment_losses).mean() if environment_losses else zero
    return classification, semantic, environment
