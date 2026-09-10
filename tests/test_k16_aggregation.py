import torch
from difter.models.aggregation import masked_mean


def test_masked_mean_shape_and_values():
    values = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    mask = torch.zeros(2, 16); mask[0, :2] = 1; mask[1, :4] = 1
    output = masked_mean(values, mask)
    assert output.shape == (2, 3)
    assert torch.allclose(output[0], values[0, :2].mean(0))
    assert torch.allclose(output[1], values[1, :4].mean(0))
