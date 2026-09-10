# Data format

Each TSV row is one traffic window. Rows are grouped by `flow_id`; `window_start` gives temporal order. A flow retains at most 16 window representation units. During training, excess windows are selected deterministically from the seed, epoch, and flow ID. During evaluation, evenly spaced temporal coverage is used. Padding repeats a placeholder row but zeros its tensors and excludes it through `window_mask`.

Required fields:

- `label`: integer class ID.
- `text_a`: whitespace-delimited traffic tokens.
- `delta_ts`: token-aligned timing values.
- `pkt_len`: token-aligned packet-length values.
- `packet_start`: token-aligned packet-boundary indicators.
- `source_file`: provenance identifier supplied by the dataset owner.
- `flow_id`: immutable flow grouping key.
- `window_start`: integer temporal order.
- one integer column per configured source-environment factor.

Real data, identities, and manifests are intentionally absent. Fit vocabulary and normalization on source-train only.
