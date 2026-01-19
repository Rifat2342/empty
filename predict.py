import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from task2_baseline.data import Task2Dataset, build_samples, load_index
from task2_baseline.model import MultiModalPoseNet
from task2_baseline.utils import denormalize_targets, parse_scenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Task 2 translations")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
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

    rows = []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            e2 = batch["e2"].to(device)
            preds = model(video, e2).cpu()
            if target_normalize:
                mean = torch.tensor(target_mean, dtype=preds.dtype)
                std = torch.tensor(target_std, dtype=preds.dtype)
                preds = denormalize_targets(preds, mean, std)

            preds_np = preds.numpy()
            for i in range(len(preds_np)):
                rows.append(
                    {
                        "scenario_id": batch["scenario_id"][i],
                        "timestamp": float(batch["timestamp"][i]),
                        "pred_x": float(preds_np[i][0]),
                        "pred_y": float(preds_np[i][1]),
                        "pred_z": float(preds_np[i][2]),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print("Wrote {} predictions".format(len(rows)))


if __name__ == "__main__":
    main()
