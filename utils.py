import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_scenarios(scenarios):
    if scenarios is None:
        return []
    if isinstance(scenarios, (list, tuple)):
        return list(scenarios)
    return [item.strip() for item in scenarios.split(",") if item.strip()]


def denormalize_targets(values, mean, std):
    if mean is None or std is None:
        return values
    return values * std + mean


def compute_regression_metrics(preds, targets, target_mean=None, target_std=None):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    if target_mean is not None and target_std is not None:
        mean = torch.tensor(target_mean, dtype=preds.dtype)
        std = torch.tensor(target_std, dtype=preds.dtype)
        preds = denormalize_targets(preds, mean, std)
        targets = denormalize_targets(targets, mean, std)

    errors = preds - targets
    mae = torch.mean(torch.abs(errors)).item()
    rmse = torch.sqrt(torch.mean(errors ** 2)).item()
    mae_axis = torch.mean(torch.abs(errors), dim=0).tolist()
    rmse_axis = torch.sqrt(torch.mean(errors ** 2, dim=0)).tolist()

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_x": mae_axis[0],
        "mae_y": mae_axis[1],
        "mae_z": mae_axis[2],
        "rmse_x": rmse_axis[0],
        "rmse_y": rmse_axis[1],
        "rmse_z": rmse_axis[2],
    }
