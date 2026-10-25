"""Summarize policy run logs (run_policy.py and run_policy_chunked.py) into one table.

Per run: checkpoint, actions per predict, time until the arm is back home after
working, number of grasps (gripper closes after having been open, while away
from home), and clamp/abort counts.

Run:
    .venv/bin/python scripts/analyze_runs.py                # all logs in logs/
    .venv/bin/python scripts/analyze_runs.py logs/x.log ... # specific logs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Mean episode start pose of the cheese dataset, motors 1-7 (raw ticks).
HOME = np.array([2056, 1020, 1979, 3076, 1817, 2343, 2458])
NEAR_HOME = 300      # sum |pos - HOME| over motors 1-7
WORKING = 1500
GRIPPER_OPEN, GRIPPER_CLOSED = 2100, 1700


def parse(path: Path) -> dict | None:
    txt = path.read_text(errors="replace")
    ckpt = re.search(r"loading (\S+) on", txt)
    chunked = re.search(r"chunking:\s+(\d+) actions", txt)
    if chunked:
        steps = re.findall(r"\[chunk \d+\] t=\s*([\d.]+)s.*?\n  pos   \[([^\]]+)\]", txt)
        t = np.array([float(s[0]) for s in steps])
        pos = np.array([[int(x) for x in s[1].split(",")] for s in steps]) if steps else np.empty((0, 8))
        clamps = len(re.findall(r"clamped [1-9]", txt))
        aborts = 0
        n = int(chunked.group(1))
    else:
        blocks = re.split(r"\n\[tick \d+\]", txt)[1:]
        rows, dts = [], []
        for b in blocks:
            r = re.findall(r"^\s+\d+\s+(\d+)\s+\d+\s+[+-]\d+", b, re.M)
            if len(r) < 8:
                continue
            rows.append([int(x) for x in r[:8]])
            # sent ticks log the full tick time; aborted ticks only fetch + predict
            m = re.search(r"total (\d+)ms", b) or re.search(r"predict (\d+)ms", b)
            dts.append(int(m.group(1)) / 1000 if m else 0.0)
        pos = np.array(rows) if rows else np.empty((0, 8))
        t = np.cumsum(dts)
        clamps = txt.count("clamped to")
        aborts = txt.count("aborted")
        n = 1
    if len(pos) == 0:
        return None

    dist = np.abs(pos[:, :7] - HOME).sum(1)
    working = np.where(dist > WORKING)[0]
    done_t = None
    if len(working):
        back = [k for k in range(working[-1], len(pos)) if dist[k] < NEAR_HOME]
        done_t = t[back[0]] if back else None

    grasps, state = 0, None
    for k, g in enumerate(pos[:, 7]):
        if g > GRIPPER_OPEN:
            state = "open"
        elif g < GRIPPER_CLOSED and state == "open":
            state = "closed"
            if dist[k] > NEAR_HOME:  # a close at home is the end of the run, not a grasp
                grasps += 1
    return {
        "log": path.name, "ckpt": ckpt.group(1) if ckpt else "?", "n": n,
        "done_s": done_t, "run_s": t[-1], "grasps": grasps, "clamps": clamps, "aborts": aborts,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", type=Path)
    args = p.parse_args()
    paths = args.logs or sorted(Path("logs").glob("*.log"), key=lambda x: x.stat().st_mtime)
    print(f"{'log':<36} {'checkpoint':<28} {'N':>3} {'done':>7} {'run':>7} {'grasps':>6} {'clamp':>5} {'abort':>5}")
    for path in paths:
        r = parse(path)
        if r is None:
            print(f"{path.name:<36} (no steps)")
            continue
        done = f"{r['done_s']:.0f}s" if r["done_s"] is not None else "-"
        print(f"{r['log']:<36} {r['ckpt']:<28} {r['n']:>3} {done:>7} {r['run_s']:>6.0f}s "
              f"{r['grasps']:>6} {r['clamps']:>5} {r['aborts']:>5}")


if __name__ == "__main__":
    main()
