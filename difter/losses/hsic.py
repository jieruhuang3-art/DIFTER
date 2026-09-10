import torch


def _rbf_kernel(value, eps=1e-6):
    distances = torch.cdist(value.float(), value.float()).square().clamp_min(0)
    eye = torch.eye(value.size(0), dtype=torch.bool, device=value.device)
    positive = distances.masked_select((~eye) & (distances > 0))
    sigma2 = positive.detach().median().clamp_min(eps) if positive.numel() else distances.new_tensor(1.0)
    return torch.exp(-distances / (2.0 * sigma2))


def _hsic(value, environment):
    count = value.size(0)
    if count < 2:
        return value.sum() * 0.0
    kernel_value = _rbf_kernel(value)
    kernel_env = environment[:, None].eq(environment[None, :]).float()
    center = torch.eye(count, device=value.device) - torch.ones(count, count, device=value.device) / count
    return torch.trace(kernel_value @ center @ kernel_env @ center) / float((count - 1) ** 2)


def class_conditional_hsic_loss(stable, labels, environment, min_samples=4, max_samples_per_class=32):
    losses = []
    for class_id in labels.unique(sorted=True):
        index = torch.nonzero(labels.eq(class_id), as_tuple=False).flatten()
        if index.numel() < min_samples or environment.index_select(0, index).unique().numel() < 2:
            continue
        if index.numel() > max_samples_per_class:
            positions = torch.linspace(0, index.numel() - 1, max_samples_per_class, device=index.device).round().long()
            index = index.index_select(0, positions)
        losses.append(_hsic(stable.index_select(0, index), environment.index_select(0, index)))
    return torch.stack(losses).mean() if losses else stable.sum() * 0.0


def cross_covariance_penalty(stable, environment):
    stable = stable.float() - stable.float().mean(0, keepdim=True)
    environment = environment.float() - environment.float().mean(0, keepdim=True)
    covariance = stable.t() @ environment / max(1, stable.size(0) - 1)
    return covariance.square().mean()
