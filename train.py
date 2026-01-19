import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task2_baseline.data import (
    Task2Dataset,
    build_samples,
    compute_e2_stats,
    compute_target_stats,
    load_index,
)
from task2_baseline.model import MultiModalPoseNet
from task2_baseline.utils import compute_regression_metrics, parse_scenarios, set_seed


def _resolve_scenarios(args, index_df):
    train_scenarios = parse_scenarios(args.train_scenarios)
    val_scenarios = parse_scenarios(args.val_scenarios)

    if args.preset == "loso":
        holdout = args.holdout_scenario
        if holdout is None:
            if len(val_scenarios) == 1:
                holdout = val_scenarios[0]
            else:
                raise ValueError(
                    "LOSO preset requires --holdout-scenario or a single --val-scenarios entry."
                )

        all_scenarios = index_df["scenario_id"].tolist()
        if holdout not in all_scenarios:
            raise ValueError(
                "Holdout scenario {} not found in dataset index.".format(holdout)
            )

        train_scenarios = [sid for sid in all_scenarios if sid != holdout]
        val_scenarios = [holdout]

    return train_scenarios, val_scenarios


def _build_loaders(args, index_df, train_scenarios, val_scenarios):
    train_samples, e2_columns = build_samples(
        index_df,
        args.dataset_root,
        train_scenarios,
        label_tolerance=args.label_tolerance,
        e2_tolerance=args.e2_tolerance,
    )

    val_samples, _ = build_samples(
        index_df,
        args.dataset_root,
        val_scenarios,
        label_tolerance=args.label_tolerance,
        e2_tolerance=args.e2_tolerance,
        e2_columns=e2_columns,
    )

    e2_mean, e2_std = compute_e2_stats(train_samples)
    target_mean, target_std = compute_target_stats(train_samples)

    train_dataset = Task2Dataset(
        train_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=args.target_normalize,
    )

    val_dataset = Task2Dataset(
        val_samples,
        image_size=args.image_size,
        video_mode=args.video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=args.target_normalize,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        e2_columns,
        e2_mean,
        e2_std,
        target_mean,
        target_std,
    )


def _run_epoch(
    model,
    loader,
    criterion,
    target_mean,
    target_std,
    optimizer=None,
    device="cpu",
    normalize_target=True,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, leave=False):
        video = batch["video"].to(device)
        e2 = batch["e2"].to(device)
        targets = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad()

        preds = model(video, e2)
        loss = criterion(preds, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_targets:
        return {"loss": float("nan"), "mae": 0.0, "rmse": 0.0}

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metric_mean = target_mean if normalize_target else None
    metric_std = target_std if normalize_target else None
    metrics = compute_regression_metrics(preds, targets, metric_mean, metric_std)
    metrics["loss"] = total_loss / targets.size(0)
    return metrics


def _build_loss(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "huber":
        return torch.nn.SmoothL1Loss()
    raise ValueError("Unknown loss: {}".format(loss_name))


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 PyTorch baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument(
        "--preset",
        choices=["none", "loso"],
        default="none",
        help="Preset data splits (loso = leave-one-scenario-out)",
    )
    parser.add_argument(
        "--holdout-scenario",
        default=None,
        help="Scenario ID to hold out when using --preset loso",
    )
    parser.add_argument(
        "--train-scenarios",
        default="exp6,exp7",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument(
        "--val-scenarios",
        default="exp8",
        help="Comma-separated list of scenario IDs",
    )
    parser.add_argument(
        "--video-mode",
        choices=["rgbd", "rgb", "disparity", "none"],
        default="rgbd",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--label-tolerance", type=float, default=0.2)
    parser.add_argument("--e2-tolerance", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs/task2_baseline")
    parser.add_argument(
        "--backbone",
        choices=["simple", "resnet18"],
        default="simple",
        help="Visual backbone for the video branch",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the visual backbone",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the visual backbone parameters",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="huber",
        help="Loss function for regression",
    )
    parser.add_argument(
        "--no-target-normalize",
        action="store_false",
        dest="target_normalize",
        help="Disable target normalization",
    )
    parser.set_defaults(target_normalize=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu or cuda). Defaults to auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.pretrained and args.backbone == "simple":
        raise ValueError("--pretrained requires --backbone resnet18")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"
    index_df = load_index(index_csv, task="task2")

    train_scenarios, val_scenarios = _resolve_scenarios(args, index_df)
    (
        train_loader,
        val_loader,
        e2_columns,
        e2_mean,
        e2_std,
        target_mean,
        target_std,
    ) = _build_loaders(args, index_df, train_scenarios, val_scenarios)

    model = MultiModalPoseNet(
        e2_dim=len(e2_columns),
        video_mode=args.video_mode,
        output_dim=3,
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = _build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            target_mean,
            target_std,
            optimizer=optimizer,
            device=device,
            normalize_target=args.target_normalize,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            target_mean,
            target_std,
            device=device,
            normalize_target=args.target_normalize,
        )

        print(
            "Epoch {:02d} | Train loss {:.4f} mae {:.2f} rmse {:.2f} | "
            "Val loss {:.4f} mae {:.2f} rmse {:.2f}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rmse"],
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rmse"],
            )
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "e2_mean": e2_mean,
            "e2_std": e2_std,
            "e2_columns": e2_columns,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_normalize": args.target_normalize,
            "video_mode": args.video_mode,
            "image_size": args.image_size,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "config": vars(args),
            "train_scenarios": train_scenarios,
            "val_scenarios": val_scenarios,
        }

        torch.save(checkpoint, out_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, out_dir / "best.pt")

    print("Best val loss: {:.4f}".format(best_val_loss))


if __name__ == "__main__":
    main()
