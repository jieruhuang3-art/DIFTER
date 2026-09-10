import argparse
from _common import runtime
from difter.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--input", required=True); parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(); _cfg, model, loaders, device = runtime(args.config, args.dataset_config)
    load_checkpoint(args.checkpoint, model, map_location=device); print("Use the configured loader and model.inference(batch) for label-free inference.")


if __name__ == "__main__": main()
