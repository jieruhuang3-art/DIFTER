"""Successful-step trainer for the progressive final objective."""

import torch
from .schedule import progressive_scales
from .checkpoint import save_checkpoint
from .prototype import PrototypeEMA


def objective_weights(config):
    return {"lambda_inv": config["ccif"]["lambda_inv"], "lambda_env": config["ccif"]["lambda_env"],
            "lambda_orth": config["ccif"]["lambda_orth"], "lambda_rec": config["ccif"]["lambda_rec"],
            "min_class_samples": config["ccif"]["min_class_samples"],
            "max_samples_per_class": config["ccif"]["max_samples_per_class"],
            "lambda_cecc": config["cecc"]["lambda"], "temperature": config["cecc"]["temperature"],
            "lambda_cei_cls": config["cei"]["lambda_cls"], "lambda_cei_sem": config["cei"]["lambda_sem"],
            "lambda_cei_env": config["cei"]["lambda_env"],
            "cosine_margin": config["training"]["cosine_margin"]}


def train(model, main_batches, support_batches, optimizer, config, device, validate, checkpoint_path,
          backbone_blocks=()):
    """Train for successful optimizer steps and select by source-validation Macro-F1."""
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=config["training"]["scaler_initial_scale"])
    weights, step, best = objective_weights(config), 0, float("-inf")
    prototype = PrototypeEMA(model.classifier.weight.size(0), model.classifier.weight.size(1), device, 0.95)
    main_iterator, support_iterator = iter(main_batches), iter(support_batches)
    while step < config["training"]["max_steps"]:
        if step >= config["training"]["freeze_steps"]:
            for parameters in backbone_blocks:
                for parameter in parameters:
                    parameter.requires_grad_(True)
        try: main_batch, main_labels, _ = next(main_iterator)
        except StopIteration: main_iterator = iter(main_batches); main_batch, main_labels, _ = next(main_iterator)
        try: support_batch, support_labels, support_environment = next(support_iterator)
        except StopIteration: support_iterator = iter(support_batches); support_batch, support_labels, support_environment = next(support_iterator)
        main_batch = {key: value.to(device) for key, value in main_batch.items()}
        support_batch = {key: value.to(device) for key, value in support_batch.items()}
        main_labels, support_labels = main_labels.to(device), support_labels.to(device)
        support_environment = {key: value.to(device) for key, value in support_environment.items()}
        optimizer.zero_grad(set_to_none=True); old_scale = scaler.get_scale(); next_step = step + 1
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model.training_objective(main_batch, main_labels, support_batch, support_labels,
                                              support_environment, progressive_scales(next_step), weights, next_step)
            if next_step >= config["training"]["freeze_steps"]:
                feature_loss = prototype.feature_loss(output["stable_main"], main_labels)
                weight_loss = prototype.weight_loss(model.classifier)
                output["loss"] = (output["loss"] + config["training"]["prototype_feature_weight"] * feature_loss
                                  + config["training"]["prototype_weight_weight"] * weight_loss)
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError("non-finite objective")
        scaler.scale(output["loss"]).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            continue
        model.classifier.clamp_scale_(); prototype.update(output["stable_main"], main_labels); step += 1
        if step % config["training"]["eval_interval"] == 0:
            metric = validate(model)
            if metric["macro_f1"] > best:
                best = metric["macro_f1"]
                save_checkpoint(checkpoint_path, model, optimizer, step, best, config)
    return {"successful_steps": step, "best_source_val_macro_f1": best}
