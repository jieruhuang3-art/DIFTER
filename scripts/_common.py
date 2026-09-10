from functools import partial
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from difter.data.preprocessing import SourceVocabulary
from difter.data.dataset import FlowDataset
from difter.data.collate import collate_flows
from difter.models.difter import DIFTER
from difter.utils.config import load_yaml


def runtime(model_config, dataset_config):
    cfg, data_cfg = load_yaml(model_config), load_yaml(dataset_config)
    train_path = data_cfg["paths"]["train"]
    import csv
    with open(train_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    vocab = SourceVocabulary().fit(rows)
    factors = data_cfg["environment_factors"]
    datasets = {name: FlowDataset(path, vocab, factors) for name, path in data_cfg["paths"].items() if Path(path).is_file()}
    cardinalities = {factor: 1 + max(flow["windows"][0]["environment"][factor] for flow in datasets["train"].flows) for factor in factors}
    model = DIFTER(data_cfg["num_classes"], cardinalities, max_windows=cfg["model"]["max_windows"])
    model.register_source_environments([
        tuple(flow["windows"][0]["environment"][factor] for factor in factors)
        for flow in datasets["train"].flows
    ])
    pretrained = cfg["model"].get("pretrained_backbone")
    if pretrained:
        payload = torch.load(pretrained, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        if any(key.startswith("backbone.") for key in state):
            backbone_state = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
        else:
            backbone_state = state
        compatible = set(model.backbone.state_dict()).intersection(backbone_state)
        model.backbone.load_state_dict(backbone_state, strict=False)
        if not any(key.startswith("encoder.transformer") for key in compatible):
            raise RuntimeError("compatible pretrained contextual encoder weights were not loaded")
    collate = partial(collate_flows, max_windows=cfg["model"]["max_windows"], max_length=cfg["model"]["max_seq_length"])
    loaders = {name: DataLoader(dataset, batch_size=cfg["training"]["flow_batch_size"], shuffle=name == "train", collate_fn=collate)
               for name, dataset in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cfg, model.to(device), loaders, device


def configure_optimizer(model, cfg):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    blocks = cfg["training"]["encoder_blocks"]
    block_lrs = cfg["training"]["encoder_block_lrs"]
    groups = []
    for index, learning_rate in zip(blocks, block_lrs):
        parameters = list(model.backbone.encoder.transformer[index].parameters())
        groups.append({"params": parameters, "lr": learning_rate})
    head_modules = [model.factorizer, model.classifier, model.cecc_projection]
    head_parameters = [parameter for module in head_modules for parameter in module.parameters()]
    for parameter in head_parameters: parameter.requires_grad_(True)
    groups.append({"params": head_parameters, "lr": cfg["training"]["head_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["training"]["weight_decay"])
    return optimizer, [group["params"] for group in groups[:-1]]
