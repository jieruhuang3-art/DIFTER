import torch
from .metrics import classification_metrics


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); truth, prediction, probabilities = [], [], []
    for batch, labels, _environment in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model.inference(batch)
        probs = logits.softmax(-1)
        truth.extend(labels.tolist()); prediction.extend(probs.argmax(-1).cpu().tolist()); probabilities.extend(probs.cpu().tolist())
    return classification_metrics(truth, prediction, probabilities)
