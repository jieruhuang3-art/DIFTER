import torch
import torch.nn.functional as F


def cecc_pair_masks(labels, environment_keys, valid_mask=None):
    valid = environment_keys.ge(0)
    if valid_mask is not None:
        valid &= valid_mask.bool()
    nonself = ~torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    positive = nonself & valid[:, None] & valid[None, :] & labels[:, None].eq(labels[None, :]) & environment_keys[:, None].ne(environment_keys[None, :])
    negative = valid[:, None] & valid[None, :] & labels[:, None].ne(labels[None, :])
    return positive, negative


def cecc_loss(projection, labels, environment_keys, temperature=0.07, valid_mask=None):
    positive, negative = cecc_pair_masks(labels, environment_keys, valid_mask)
    valid_anchor = positive.any(1)
    if not valid_anchor.any():
        return projection.sum() * 0.0
    similarity = F.normalize(projection, dim=-1).float() @ F.normalize(projection, dim=-1).float().t()
    logits = similarity / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    candidate = positive | negative
    denominator = torch.logsumexp(logits.masked_fill(~candidate, float("-inf")), 1)
    log_probability = logits - denominator[:, None]
    per_anchor = -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))
    return per_anchor[valid_anchor].mean()
