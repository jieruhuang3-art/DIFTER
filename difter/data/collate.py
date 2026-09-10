import torch
from .traffic_windows import select_windows


def collate_flows(flows, max_windows=16, max_length=128, epoch=0, training=False):
    tensors = {name: [] for name in ("token_ids", "segments", "delta_ts", "pkt_len", "packet_start")}
    window_masks, labels, environment = [], [], {}
    for flow in flows:
        windows, mask = select_windows(flow, max_windows, epoch, training)
        window_masks.append(mask); labels.append(flow["label"])
        for factor in windows[0]["environment"]:
            environment.setdefault(factor, []).append(windows[0]["environment"][factor])
        for window, valid in zip(windows, mask):
            length = min(len(window["token_ids"]), max_length)
            pad = max_length - length
            tensors["token_ids"].append(window["token_ids"][:length] + [0] * pad)
            tensors["segments"].append(([1] * length + [0] * pad) if valid else [0] * max_length)
            for source, target, cast in (("delta_ts", "delta_ts", float), ("pkt_len", "pkt_len", float), ("packet_start", "packet_start", int)):
                values = [cast(value) for value in window[source]][:length] + [cast(0)] * pad
                tensors[target].append(values if valid else [cast(0)] * max_length)
    batch = {name: torch.tensor(values, dtype=torch.long if name in {"token_ids", "segments", "packet_start"} else torch.float32)
             .reshape(len(flows), max_windows, max_length) for name, values in tensors.items()}
    batch["window_mask"] = torch.tensor(window_masks, dtype=torch.float32)
    return batch, torch.tensor(labels, dtype=torch.long), {name: torch.tensor(values, dtype=torch.long) for name, values in environment.items()}
