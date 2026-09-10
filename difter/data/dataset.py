import csv
from collections import defaultdict
from pathlib import Path


def _numbers(value, cast=float):
    return [cast(item) for item in value.replace(",", " ").split()]


class FlowDataset:
    """Group processed traffic-window TSV rows by immutable flow identifier."""

    REQUIRED = {"label", "text_a", "delta_ts", "pkt_len", "packet_start", "flow_id", "window_start"}

    def __init__(self, path, vocabulary, environment_factors):
        grouped = defaultdict(list)
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = self.REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            for row in reader:
                token_ids = vocabulary.encode(row["text_a"])
                window = {"token_ids": token_ids, "delta_ts": _numbers(row["delta_ts"]),
                          "pkt_len": _numbers(row["pkt_len"]), "packet_start": _numbers(row["packet_start"], int),
                          "window_start": int(row["window_start"]),
                          "environment": {factor: int(row[factor]) for factor in environment_factors}}
                grouped[row["flow_id"]].append((row["label"], window))
        self.flows = []
        for flow_id, rows in grouped.items():
            labels = {label for label, _ in rows}
            if len(labels) != 1:
                raise ValueError(f"inconsistent labels for flow {flow_id}")
            windows = [window for _, window in rows]
            windows.sort(key=lambda item: item["window_start"])
            self.flows.append({"flow_id": flow_id, "label": int(next(iter(labels))), "windows": windows})
        self.flows.sort(key=lambda item: item["flow_id"])

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        return self.flows[index]
