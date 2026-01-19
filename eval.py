import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from task2_baseline.data import Task2Dataset, build_samples, load_index
from task2_baseline.model import MultiModalPoseNet
from task2_baseline.utils import compute_regression_metrics, parse_scenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 baseline")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.scenarios = parse_scenarios(args.scenarios)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    e2_columns = checkpoint["e2_columns"]
    e2_mean = checkpoint["e2_mean"]
    e2_std = checkpoint["e2_std"]
    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")
    target_normalize = checkpoint.get("target_normalize", True)
    video_mode = checkpoint.get("video_mode", "rgbd")
    image_size = checkpoint.get("image_size", 128)
    backbone = checkpoint.get(
        "backbone", checkpoint.get("config", {}).get("backbone", "simple")
    )

    index_csv = args.index_csv
    if index_csv is None:
        index_csv = Path(args.dataset_root) / "index.csv"

    index_df = load_index(index_csv, task="task2")
    samples, _ = build_samples(
        index_df,
        args.dataset_root,
        args.scenarios,
        label_tolerance=checkpoint["config"].get("label_tolerance", 0.2),
        e2_tolerance=checkpoint["config"].get("e2_tolerance", 0.05),
        e2_columns=e2_columns,
    )

    dataset = Task2Dataset(
        samples,
        image_size=image_size,
        video_mode=video_mode,
        e2_mean=e2_mean,
        e2_std=e2_std,
        target_mean=target_mean,
        target_std=target_std,
        normalize_target=target_normalize,
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiModalPoseNet(
        e2_dim=len(e2_columns),
        video_mode=video_mode,
        output_dim=3,
        backbone=backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            e2 = batch["e2"].to(device)
            targets = batch["target"].to(device)
            preds = model(video, e2)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metric_mean = target_mean if target_normalize else None
    metric_std = target_std if target_normalize else None
    metrics = compute_regression_metrics(preds, targets, metric_mean, metric_std)

    print("MAE (mm): {:.2f}".format(metrics["mae"]))
    print("RMSE (mm): {:.2f}".format(metrics["rmse"]))
    print(
        "Per-axis MAE (mm): x={:.2f} y={:.2f} z={:.2f}".format(
            metrics["mae_x"], metrics["mae_y"], metrics["mae_z"]
        )
    )
    print(
        "Per-axis RMSE (mm): x={:.2f} y={:.2f} z={:.2f}".format(
            metrics["rmse_x"], metrics["rmse_y"], metrics["rmse_z"]
        )
    )


if __name__ == "__main__":
    main()
