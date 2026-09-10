import torch
from difter.losses.contrastive import cecc_pair_masks


def test_cecc_pair_definitions():
    labels = torch.tensor([0, 0, 0, 1])
    environments = torch.tensor([0, 1, 0, 0])
    positive, negative = cecc_pair_masks(labels, environments)
    assert positive[0, 1] and positive[1, 0]
    assert not positive[0, 2]
    assert negative[0, 3] and negative[3, 2]
    assert not negative[0, 1]
