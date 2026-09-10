import torch
from difter.models.hptf import HPTFEncoder


def test_hptf_window_shape():
    model = HPTFEncoder(vocab_size=64, max_seq_length=8, hidden_size=24, heads=4,
                        feedforward_size=48, context_layers=1, intra_packet_layers=1,
                        inter_packet_layers=1, field_slots=4, dropout=0.0)
    token = torch.randint(0, 64, (2, 8)); segments = torch.ones(2, 8, dtype=torch.long)
    timing = torch.rand(2, 8); length = torch.rand(2, 8)
    starts = torch.zeros(2, 8, dtype=torch.long); starts[:, [0, 4]] = 1
    assert model(token, segments, timing, length, starts).shape == (2, 24)
