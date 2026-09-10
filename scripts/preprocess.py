import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    required = {"label", "text_a", "delta_ts", "pkt_len", "packet_start", "source_file", "flow_id", "window_start"}
    import csv
    for split in ("train", "dev"):
        path = Path(config["paths"][split])
        with path.open(encoding="utf-8", newline="") as handle:
            fields = set(next(csv.reader(handle, delimiter="\t")))
        missing = required - fields
        if missing: raise ValueError(f"{split} missing fields: {sorted(missing)}")
    print("Source data schema validated; no target data was read.")


if __name__ == "__main__": main()
