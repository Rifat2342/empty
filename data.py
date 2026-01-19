import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_index(index_csv, task="task2"):
    index_df = pd.read_csv(index_csv)
    index_df = index_df[index_df["task"] == task].reset_index(drop=True)
    return index_df


def parse_scenarios(scenarios):
    if scenarios is None:
        return []
    if isinstance(scenarios, (list, tuple)):
        return list(scenarios)
    return [item.strip() for item in scenarios.split(",") if item.strip()]


def _resample_mode():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def _load_color(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), _resample_mode())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_disparity(path, image_size):
    img = Image.open(path).convert("L")
    img = img.resize((image_size, image_size), _resample_mode())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _validate_e2_columns(columns, reference_columns):
    missing = [col for col in reference_columns if col not in columns]
    if missing:
        raise ValueError(
            "E2 feature columns mismatch. Missing columns: {}".format(", ".join(missing)
            )
        )


def build_samples(
    index_df,
    dataset_root,
    scenarios,
    label_tolerance=0.2,
    e2_tolerance=0.05,
    e2_columns=None,
):
    dataset_root = Path(dataset_root)
    scenarios = set(parse_scenarios(scenarios))
    if scenarios:
        scenario_df = index_df[index_df["scenario_id"].isin(scenarios)]
    else:
        scenario_df = index_df

    if scenario_df.empty:
        raise ValueError("No scenarios matched the provided list.")

    samples = []
    reference_e2_columns = list(e2_columns) if e2_columns else None

    for row in scenario_df.itertuples(index=False):
        frames_path = dataset_root / row.video_frames_csv
        video_root = frames_path.parent
        frames_df = pd.read_csv(frames_path)
        frames_df = frames_df.rename(columns={"timestamp": "frame_time"})
        frames_df = frames_df.sort_values("frame_time")

        ann_path = dataset_root / row.annotation
        ann_df = pd.read_csv(ann_path)
        ann_df = ann_df.rename(columns={"timestamp": "label_time"})
        ann_df = ann_df.sort_values("label_time")

        aligned = pd.merge_asof(
            frames_df.sort_values("frame_time"),
            ann_df[["label_time", "x", "y", "z"]].sort_values("label_time"),
            left_on="frame_time",
            right_on="label_time",
            direction="nearest",
            tolerance=label_tolerance,
        )
        aligned = aligned.dropna(subset=["x", "y", "z"])

        e2_path = dataset_root / row.radio_e2
        e2_df = pd.read_csv(e2_path)
        e2_df["timestamp"] = e2_df["timestamp"].astype(float)
        e2_df = e2_df.sort_values("timestamp")
        numeric_cols = e2_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "timestamp"]

        if reference_e2_columns is None:
            reference_e2_columns = numeric_cols
        else:
            _validate_e2_columns(numeric_cols, reference_e2_columns)

        aligned = pd.merge_asof(
            aligned.sort_values("frame_time"),
            e2_df[["timestamp"] + reference_e2_columns].sort_values("timestamp"),
            left_on="frame_time",
            right_on="timestamp",
            direction="nearest",
            tolerance=e2_tolerance,
        )
        aligned = aligned.dropna(subset=reference_e2_columns)

        for item in aligned.itertuples(index=False):
            item_dict = item._asdict()
            e2_vals = np.array(
                [item_dict[col] for col in reference_e2_columns], dtype=np.float32
            )
            target = np.array(
                [item_dict["x"], item_dict["y"], item_dict["z"]], dtype=np.float32
            )
            samples.append(
                {
                    "color_path": str(video_root / item_dict["color"]),
                    "disparity_path": str(video_root / item_dict["disparity"]),
                    "e2": e2_vals,
                    "target": target,
                    "timestamp": float(item_dict["frame_time"]),
                    "scenario_id": row.scenario_id,
                }
            )

    return samples, reference_e2_columns


def compute_e2_stats(samples):
    features = np.stack([sample["e2"] for sample in samples], axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_target_stats(samples):
    targets = np.stack([sample["target"] for sample in samples], axis=0)
    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class Task2Dataset(Dataset):
    def __init__(
        self,
        samples,
        image_size=128,
        video_mode="rgbd",
        e2_mean=None,
        e2_std=None,
        target_mean=None,
        target_std=None,
        normalize_target=True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.video_mode = video_mode
        self.use_video = video_mode != "none"
        self.use_color = video_mode in ("rgb", "rgbd")
        self.use_disparity = video_mode in ("disparity", "rgbd")
        self.e2_mean = e2_mean
        self.e2_std = e2_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.normalize_target = normalize_target

        if video_mode == "rgbd":
            self.video_channels = 4
        elif video_mode == "rgb":
            self.video_channels = 3
        elif video_mode == "disparity":
            self.video_channels = 1
        else:
            self.video_channels = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        e2 = sample["e2"].astype(np.float32)
        if self.e2_mean is not None and self.e2_std is not None:
            e2 = (e2 - self.e2_mean) / self.e2_std
        e2_tensor = torch.from_numpy(e2)

        target = sample["target"].astype(np.float32)
        if self.normalize_target and self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std
        target_tensor = torch.from_numpy(target)

        if self.use_video:
            channels = []
            if self.use_color:
                channels.append(_load_color(sample["color_path"], self.image_size))
            if self.use_disparity:
                channels.append(
                    _load_disparity(sample["disparity_path"], self.image_size)
                )
            if channels:
                video = torch.cat(channels, dim=0)
            else:
                video = torch.zeros(
                    (self.video_channels, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
        else:
            video = torch.zeros(
                (self.video_channels, self.image_size, self.image_size),
                dtype=torch.float32,
            )

        return {
            "video": video,
            "e2": e2_tensor,
            "target": target_tensor,
            "timestamp": sample["timestamp"],
            "scenario_id": sample["scenario_id"],
        }


def save_metadata(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
