"""Check exported norma-core episodes (parquet) for training usability.

Each episode gets OK / WARN / FAIL with the reasons:

    FAIL  structure broken (joint/camera count, unreadable JPEG, norm values outside [0, 1]),
          frame gap > --max-gap-s, shorter than --min-duration-s, wrong task text,
          arm never left home, no grasp at all
    WARN  fps off, longer than --max-duration-s, fewer grasps than --min-grasps,
          camera frozen (same JPEG > 1 s), arm idle > --max-idle-s, start not at home

Episodes are grouped by `episode_start_ns`, so per-episode files and merged files both work.
Exit code 1 if any episode FAILs.

Run (from smolvla_py):
    .venv/bin/python scripts/check_episodes.py ~/datasets/dataset-rc2-ep*.parquet
    .venv/bin/python scripts/check_episodes.py ~/datasets/dataset-rc2.parquet \
        --task "Spread the cheese relatively evenly over the pizza." --min-grasps 4
"""

from __future__ import annotations

import argparse
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Mean episode start pose of the cheese dataset, motors 1-7 (raw ticks).
HOME = np.array([2056, 1020, 1979, 3076, 1817, 2343, 2458])
GRIPPER_OPEN, GRIPPER_CLOSED = 1950, 1850  # same as make_stage_dataset.py / planner.py
DROP_MIN_DISTANCE = 600
# A retry (grip missed, gripper reopened at the bowl) moves the arm ~200 ticks between close and
# open; a real drop carries the piece bowl -> pizza (>= ~470 in rc4). Only the latter counts.
DROP_MIN_MOVE = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquets", nargs="+", help="parquet files (shell globs OK)")
    p.add_argument("--task", help="expected task text; mismatch = FAIL")
    p.add_argument("--joints", type=int, default=8)
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--max-gap-s", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=10.0)
    p.add_argument("--max-duration-s", type=float, default=180.0)
    p.add_argument("--min-grasps", type=int, default=1)
    p.add_argument("--drops", type=int, default=0,
                   help="expected drops per episode (make_stage_dataset.py rule); mismatch = FAIL. 0 = no check")
    p.add_argument("--max-idle-s", type=float, default=8.0)
    p.add_argument("--home", help="home pose motors 1-7 as comma list (default: cheese-v1 start pose); "
                                  "rc3 rest pose: 2047,1016,964,3149,1886,2983,2197")
    p.add_argument("--home-distance", type=int, default=600, help="sum |pos - home| (motors 1-7) that counts as home")
    p.add_argument("--work-distance", type=int, default=1500)
    p.add_argument("--image-sample", type=int, default=20, help="decode every Nth JPEG (speed)")
    p.add_argument("-v", "--verbose", action="store_true", help="also list OK details")
    return p.parse_args()


def count_grasps(gripper: np.ndarray, dist: np.ndarray, home_distance: int) -> int:
    """Gripper closes after having been open, while away from home."""
    grasps, state = 0, None
    for g, d in zip(gripper, dist):
        if g > GRIPPER_OPEN:
            state = "open"
        elif g < GRIPPER_CLOSED and state == "open":
            state = "closed"
            if d > home_distance:
                grasps += 1
    return grasps


def longest_run(mask: np.ndarray, t: np.ndarray) -> float:
    """Longest contiguous time span where mask is True."""
    best, start = 0.0, None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            best = max(best, t[end] - t[start])
            start = None
    return best


def check_episode(rows: dict, args) -> tuple[str, list[str], dict]:
    fails, warns = [], []
    t = np.array(rows["timestamp_ns_since_episode_start"], dtype=np.float64) / 1e9
    joints = rows["joints"]
    images = rows["images"]
    n = len(t)

    # --- structure ---
    n_joints = {len(j) for j in joints}
    n_cams = {len(im) for im in images}
    if n_joints != {args.joints}:
        fails.append(f"joints per frame {sorted(n_joints)} != {args.joints}")
    if n_cams != {args.cameras}:
        fails.append(f"cameras per frame {sorted(n_cams)} != {args.cameras}")
    if fails:
        return "FAIL", fails, {"frames": n}
    pos = np.array([[m["position"] for m in j] for j in joints])
    pos_norm = np.array([[m["position_norm"] for m in j] for j in joints])
    goal_norm = np.array([[m["goal_norm"] for m in j] for j in joints])
    for name, arr in (("position_norm", pos_norm), ("goal_norm", goal_norm)):
        if not np.isfinite(arr).all() or arr.min() < -1e-3 or arr.max() > 1 + 1e-3:
            fails.append(f"{name} outside [0, 1] or NaN")
    bad_jpeg = 0
    for i in range(0, n, max(1, args.image_sample)):
        for cam in images[i]:
            try:
                with Image.open(io.BytesIO(cam["jpeg"])) as im:
                    im.load()
            except Exception:
                bad_jpeg += 1
    if bad_jpeg:
        fails.append(f"{bad_jpeg} unreadable JPEG(s) in sample")

    # --- timing ---
    duration = t[-1] if n else 0.0
    dts = np.diff(t)
    max_gap = dts.max() if len(dts) else 0.0
    fps = (n - 1) / duration if duration > 0 else 0.0
    if max_gap > args.max_gap_s:
        fails.append(f"frame gap {max_gap:.2f}s")
    if duration < args.min_duration_s:
        fails.append(f"too short ({duration:.1f}s)")
    elif duration > args.max_duration_s:
        warns.append(f"long ({duration:.0f}s > {args.max_duration_s:.0f}s eval timeout)")
    if not 9.0 <= fps <= 11.0:
        warns.append(f"fps {fps:.1f}")

    # --- task ---
    tasks = set(rows["task"])
    if args.task and tasks != {args.task}:
        fails.append(f"task {sorted(tasks)} != expected")

    # --- behaviour ---
    dist = np.abs(pos[:, :7] - HOME).sum(1)
    grasps = count_grasps(pos[:, 7], dist, args.home_distance)
    if dist.max() < args.work_distance:
        fails.append("arm never left home")
    if grasps == 0:
        fails.append("no grasp (gripper never opened then closed away from home)")
    elif grasps < args.min_grasps:
        warns.append(f"only {grasps} grasp(s) < {args.min_grasps}")
    drops, state, close_pose = 0, None, None  # same rule as make_stage_dataset.find_drops
    for p_, g, dd in zip(pos[:, :7], pos[:, 7], dist):
        if g < GRIPPER_CLOSED and state != "closed":
            state, close_pose = "closed", p_
        elif g > GRIPPER_OPEN and state == "closed":
            state = "open"
            drops += bool(dd > DROP_MIN_DISTANCE and np.abs(p_ - close_pose).sum() > DROP_MIN_MOVE)
    if args.drops and drops != args.drops:
        fails.append(f"{drops} drops detected, expected {args.drops} (stage split would skip it)")
    if dist[0] > args.home_distance:
        warns.append(f"start not at home (distance {int(dist[0])})")

    # camera frozen: identical JPEG bytes on consecutive frames
    for cam in range(args.cameras):
        same = np.array([False] + [images[i][cam]["jpeg"] == images[i - 1][cam]["jpeg"] for i in range(1, n)])
        frozen = longest_run(same, t)
        if frozen > 1.0:
            warns.append(f"cam{cam} frozen {frozen:.1f}s")

    # arm idle: all joints (incl. gripper) move < 5 ticks frame to frame
    still = np.r_[False, (np.abs(np.diff(pos, axis=0)) < 5).all(1)]
    idle = longest_run(still, t)
    if idle > args.max_idle_s:
        warns.append(f"arm idle {idle:.0f}s")

    status = "FAIL" if fails else "WARN" if warns else "OK"
    info = {"frames": n, "duration": duration, "grasps": grasps, "max_gap": max_gap,
            "end_home": dist[-1] <= args.home_distance}
    return status, fails + warns, info


def main() -> None:
    global HOME
    args = parse_args()
    if args.home:
        HOME = np.array([int(x) for x in args.home.split(",")][:7])
    files = sorted({f for pat in args.parquets for f in (glob.glob(str(Path(pat).expanduser())) or [pat])})
    cols = ["episode_start_ns", "timestamp_ns_since_episode_start", "joints", "images", "task"]
    print(f"{'file':<34} {'ep':>3} {'status':<5} {'frames':>6} {'dur':>6} {'grasps':>6} {'gap':>5} {'end':>5}  notes")
    totals = {"OK": 0, "WARN": 0, "FAIL": 0}
    for f in files:
        table = pq.read_table(f, columns=cols)
        ep_ids = table.column("episode_start_ns").to_pylist()
        order = list(dict.fromkeys(ep_ids))
        data = {c: table.column(c).to_pylist() for c in cols[1:]}
        for k, ep in enumerate(order, 1):
            idx = [i for i, e in enumerate(ep_ids) if e == ep]
            rows = {c: [data[c][i] for i in idx] for c in data}
            status, notes, info = check_episode(rows, args)
            totals[status] += 1
            end = ("home" if info.get("end_home") else "away") if "end_home" in info else "-"
            print(f"{Path(f).name[-34:]:<34} {k:>3} {status:<5} {info['frames']:>6} "
                  f"{info.get('duration', 0):>5.0f}s {info.get('grasps', 0):>6} {info.get('max_gap', 0):>4.2f}s {end:>5}  "
                  + ("; ".join(notes) if notes or args.verbose else ""))
    total = sum(totals.values())
    print(f"\n{total} episode(s): {totals['OK']} OK, {totals['WARN']} WARN, {totals['FAIL']} FAIL"
          + ("  -> drop FAIL episodes before merging" if totals["FAIL"] else ""))
    sys.exit(1 if totals["FAIL"] else 0)


if __name__ == "__main__":
    main()
