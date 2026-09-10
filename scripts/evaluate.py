import argparse
from _common import runtime
from difter.evaluation.evaluator import evaluate
from difter.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--split", choices=["source_test", "target_test"], required=True); parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(); _cfg, model, loaders, device = runtime(args.config, args.dataset_config)
    load_checkpoint(args.checkpoint, model, map_location=device); print(evaluate(model, loaders[args.split], device))


if __name__ == "__main__": main()
