import argparse
from pathlib import Path
from _common import runtime, configure_optimizer
from difter.evaluation.evaluator import evaluate
from difter.training.trainer import train
from difter.data.sampler import MainFlowBatchIterator, SupportFlowBatchIterator


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args(); cfg, model, loaders, device = runtime(args.config, args.dataset_config)
    if "dev" not in loaders:
        raise ValueError("source-validation path is required")
    optimizer, backbone_blocks = configure_optimizer(model, cfg)
    validate = lambda current: evaluate(current, loaders["dev"], device)
    train_flows = loaders["train"].dataset.flows
    main_batches = MainFlowBatchIterator(train_flows, cfg["model"]["max_windows"], cfg["model"]["max_seq_length"])
    factor = next(iter(cfg.get("environment_factors", [])), None)
    if factor is None:
        from difter.utils.config import load_yaml
        factor = load_yaml(args.dataset_config)["environment_factors"][0]
    support_batches = SupportFlowBatchIterator(train_flows, factor, cfg["model"]["max_windows"], cfg["model"]["max_seq_length"])
    Path("checkpoints").mkdir(exist_ok=True)
    print(train(model, main_batches, support_batches, optimizer, cfg, device, validate,
                "checkpoints/best.pt", backbone_blocks))


if __name__ == "__main__": main()
