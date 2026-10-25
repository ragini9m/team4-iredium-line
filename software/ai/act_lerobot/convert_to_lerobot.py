"""Convert norma-core dataset parquets into a local LeRobotDataset (v3) for lerobot-train.

Mapping (same as smolvla_py, so a policy can be deployed the same way):
    observation.state            joints[*].position_norm   (n_joints,) float32
    action                       joints[*].goal_norm       (n_joints,) float32
    observation.images.cam0/1    images[0/1].jpeg          224x224 RGB, stored as video
    task                         task

Episodes are split on `episode_start_ns`. Frames are ~10 Hz in norma-core datasets.

Run with the lerobot environment:
    ~/hack26/lerobot/.venv/bin/python convert_to_lerobot.py \
        --parquets ~/datasets/dataset-cheese.parquet \
        --repo-id local/cheese_v1 --root ~/datasets/lerobot/cheese_v1
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERAS = ("cam0", "cam1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquets", type=Path, nargs="+", required=True)
    p.add_argument("--repo-id", required=True, help="e.g. local/cheese_v1 (only used as a name)")
    p.add_argument("--root", type=Path, required=True, help="Output directory of the LeRobotDataset")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    first = pq.read_table(args.parquets[0]).slice(0, 1).to_pylist()[0]
    n_joints = len(first["joints"])
    height, width, _ = decode(first["images"][0]["jpeg"]).shape
    joint_names = [f"motor_{i + 1}" for i in range(n_joints)]

    features = {
        "observation.state": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"],
        }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, root=args.root,
        robot_type="normacore_st3215", use_videos=True,
    )

    n_episodes = n_frames = 0
    for path in args.parquets:
        table = pq.read_table(path)
        episode_ids = table.column("episode_start_ns").to_pylist()
        rows = table.to_pylist()
        for i, row in enumerate(rows):
            ds.add_frame({
                "observation.state": np.array([j["position_norm"] for j in row["joints"]], dtype=np.float32),
                "action": np.array([j["goal_norm"] for j in row["joints"]], dtype=np.float32),
                **{f"observation.images.{cam}": decode(row["images"][k]["jpeg"]) for k, cam in enumerate(CAMERAS)},
                "task": row["task"],
            })
            n_frames += 1
            if i + 1 == len(rows) or episode_ids[i + 1] != episode_ids[i]:
                ds.save_episode()
                n_episodes += 1
                print(f"episode {n_episodes} saved ({n_frames} frames total)")
    ds.finalize()
    print(f"\ndone: {n_episodes} episodes, {n_frames} frames, {n_joints} joints, "
          f"{len(CAMERAS)} cameras {width}x{height} @ {args.fps} fps -> {args.root}")


if __name__ == "__main__":
    main()
