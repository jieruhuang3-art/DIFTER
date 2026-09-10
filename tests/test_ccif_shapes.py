import torch
from difter.models.ccif import CCIF


def test_ccif_shapes():
    model = CCIF({"capture": 3, "device": 2})
    output = model(torch.randn(7, 768))
    assert output["stable"].shape == (7, 256)
    assert output["environment"]["capture"].shape == (7, 128)
    assert output["environment_logits"]["device"].shape == (7, 2)
    assert output["reconstruction"].shape == (7, 768)
