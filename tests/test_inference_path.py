import torch
import torch.nn as nn
from difter.models.difter import DIFTER
from difter.training.schedule import progressive_scales


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(32, 16); self.projection = nn.Linear(16, 16)
    def forward(self, token_ids, segments, delta_ts, pkt_len, packet_start):
        mask = segments.gt(0).unsqueeze(-1)
        value = (self.embedding(token_ids) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.projection(value)


def batch(size=4, windows=2, length=6):
    return {"token_ids": torch.randint(0, 32, (size, windows, length)),
            "segments": torch.ones(size, windows, length, dtype=torch.long),
            "delta_ts": torch.rand(size, windows, length), "pkt_len": torch.rand(size, windows, length),
            "packet_start": torch.zeros(size, windows, length, dtype=torch.long),
            "window_mask": torch.ones(size, windows)}


def model():
    return DIFTER(2, {"capture": 2}, backbone=TinyBackbone(), max_windows=2, feature_dim=16, stable_dim=256)


def test_training_forward_is_finite():
    current = model(); labels = torch.tensor([0, 0, 1, 1]); environment = {"capture": torch.tensor([0, 1, 0, 1])}
    weights = {"lambda_inv": .1, "lambda_env": .2, "lambda_orth": .05, "lambda_rec": .1,
               "min_class_samples": 2, "max_samples_per_class": 32, "lambda_cecc": .2,
               "temperature": .07, "lambda_cei_cls": .2, "lambda_cei_sem": .1,
               "lambda_cei_env": .05, "cosine_margin": .1}
    output = current.training_objective(batch(), labels, batch(), labels, environment, progressive_scales(4000), weights)
    assert torch.isfinite(output["loss"])


def test_inference_uses_only_stable_path_and_no_environment_labels():
    current = model()
    class Forbidden(nn.Module):
        def forward(self, *args, **kwargs): raise AssertionError("training-only module called")
    current.cecc_projection = Forbidden(); current.cei = Forbidden()
    logits = current.inference(batch())
    assert logits.shape == (4, 2)
