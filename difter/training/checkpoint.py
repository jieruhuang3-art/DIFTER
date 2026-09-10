from pathlib import Path
import os
import torch


def save_checkpoint(path, model, optimizer, step, best_metric, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "best_source_val_macro_f1": best_metric, "config": config}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
