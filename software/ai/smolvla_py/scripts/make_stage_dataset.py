"""Split each demo episode at its cheese drops into per-stage episodes with stage prompts.

SmolVLA has no memory: with one prompt for the whole routine it cannot tell drop 1
from drop 4 and repeats the first one. Here every demo is cut at its drops (gripper
closes on a grasp, then opens again away from home) into `--stages` segments, each
with its own prompt, e.g. "Put cheese on spot 3 of 7." The planner then runs the
stages in order and switches prompt when it detects the drop (same rule as here).

    segment k (k < n):  previous drop  →  drop k + --tail-s   (overlaps the next stage a bit)
    segment n:          drop n-1       →  end of episode      (last drop + return home)

Episodes whose drop count differs from --stages are skipped (reported).

Run (from smolvla_py):
    .venv/bin/python scripts/make_stage_dataset.py "$HOME/datasets/dataset-rc3-ep*.parquet" \
        --output ~/datasets/dataset-rc3-staged.parquet \
        --home 2047,1016,964,3149,1886,2983,2197
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Shared with planner.py / check_episodes.py. Grasps in rc3 reach ~1450-1600 but cross 1700 only
# barely, so "closed" < 1850 leaves margin; drops count from 600 away from home (some pizza spots are
# within 1500 of the rest pose; the rest pose has the gripper open, so no close/open cycle happens there).
GRIPPER_OPEN, GRIPPER_CLOSED = 1950, 1850
DROP_MIN_DISTANCE = 600
# A retry (grip missed, gripper reopened at the bowl) moves the arm ~200 ticks between close and
# open; a real drop carries the piece bowl -> pizza (>= ~470 in rc4). Only the latter counts.
DROP_MIN_MOVE = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquets", nargs="+", help="per-episode or merged parquets (shell globs OK)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", type=int, default=7)
    p.add_argument("--prompt", default="Put cheese on spot {k} of {n}.")
    p.add_argument("--tail-s", type=float, default=1.0, help="frames kept after each drop in that stage")
    p.add_argument("--home", required=True, help="home pose motors 1-7, comma list")
    p.add_argument("--work-distance", type=int, default=DROP_MIN_DISTANCE,
                   help="sum |pos - home| (motors 1-7) a drop must be away from home")
    p.add_argument("--gripper-index", type=int, default=7)
    return p.parse_args()


def find_drops(pos: np.ndarray, home: np.ndarray, work_distance: int, gi: int) -> list[int]:
    """Frame indices where the gripper opens after having closed, away from home, with the arm
    moved since closing (a reopen at the bowl after a missed grip is a retry, not a drop)."""
    drops, state, close_pose = [], None, None
    dist = np.abs(pos[:, :7] - home).sum(1)
    for i, g in enumerate(pos[:, gi]):
        if g < GRIPPER_CLOSED and state != "closed":
            state, close_pose = "closed", pos[i, :7]
        elif g > GRIPPER_OPEN and state == "closed":
            state = "open"
            if dist[i] > work_distance and np.abs(pos[i, :7] - close_pose).sum() > DROP_MIN_MOVE:
                drops.append(i)
    return drops


def main() -> None:
    args = parse_args()
    home = np.array([int(x) for x in args.home.split(",")][:7])
    files = sorted({f for pat in args.parquets for f in (glob.glob(str(Path(pat).expanduser())) or [pat])})
    out_tables, skipped, n_eps = [], [], 0
    for f in files:
        table = pq.read_table(f)
        ep_col = np.array(table.column("episode_start_ns").to_pylist(), dtype=np.uint64)
        for ep in dict.fromkeys(ep_col.tolist()):
            rows = np.flatnonzero(ep_col == ep)
            ep_table = table.take(pa.array(rows))
            pos = np.array([[m["position"] for m in j] for j in ep_table.column("joints").to_pylist()])
            t = np.array(ep_table.column("timestamp_ns_since_episode_start").to_pylist(), dtype=np.uint64)
            drops = find_drops(pos, home, args.work_distance, args.gripper_index)
            if len(drops) != args.stages:
                skipped.append(f"{Path(f).name} ep@{ep}: {len(drops)} drops")
                continue
            n_eps += 1
            tail = int(round(args.tail_s * (len(t) - 1) / (t[-1] / 1e9)))
            starts = [0] + drops[:-1]
            ends = [min(d + tail, len(t)) for d in drops[:-1]] + [len(t)]
            for k, (a, b) in enumerate(zip(starts, ends), 1):
                seg = ep_table.slice(a, b - a)
                n = seg.num_rows
                seg = seg.set_column(seg.schema.get_field_index("episode_start_ns"), "episode_start_ns",
                                     pa.array([int(ep) + k] * n, type=pa.uint64()))
                seg = seg.set_column(seg.schema.get_field_index("timestamp_ns_since_episode_start"),
                                     "timestamp_ns_since_episode_start",
                                     pa.array((t[a:b] - t[a]).tolist(), type=pa.uint64()))
                seg = seg.set_column(seg.schema.get_field_index("task"), "task",
                                     pa.array([args.prompt.format(k=k, n=args.stages)] * n, type=pa.string()))
                out_tables.append(seg)
    if not out_tables:
        raise SystemExit("no episode had the expected number of drops")
    result = pa.concat_tables(out_tables)
    pq.write_table(result, args.output, compression="snappy")
    lengths = [t.num_rows for t in out_tables]
    print(f"{n_eps} episodes → {len(out_tables)} stage segments, {result.num_rows} rows → {args.output}")
    print(f"segment length: min {min(lengths)}, median {int(np.median(lengths))}, max {max(lengths)} frames")
    for s in skipped:
        print(f"  skipped: {s}")


if __name__ == "__main__":
    main()
