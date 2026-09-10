import hashlib
import random
import numpy as np


def select_windows(flow, k=16, epoch=0, training=False):
    """Exact flow-window policy: seeded training subset; temporal evaluation coverage."""
    values = flow["windows"]
    if not values:
        raise ValueError("flow has no windows")
    if len(values) <= k:
        selected = list(values)
    elif training:
        seed = int.from_bytes(hashlib.sha256(f"42:{epoch}:{flow['flow_id']}".encode()).digest()[:8], "little")
        indices = sorted(random.Random(seed).sample(range(len(values)), k))
        selected = [values[index] for index in indices]
    else:
        indices = np.linspace(0, len(values) - 1, k).round().astype(int)
        selected = [values[int(index)] for index in indices]
    mask = [1.0] * len(selected)
    while len(selected) < k:
        selected.append(selected[-1])
        mask.append(0.0)
    return selected, mask
