"""Continuous policy loop with action chunking.

Same model, frame handling and commands as `run_policy.py`, but instead of
executing only the first action of every predicted chunk, it executes the first
`--actions-per-predict` actions at `--fps` (training data is ~10 Hz) before
predicting again. Closer to the demonstration speed and smoother motion.

Per chunk:
  1. Fetch a fresh Frame from `inference/normvla`.
  2. Predict a chunk (50 actions) → unnormalize → raw servo ticks.
  3. Send actions 0..N-1, one sync-write every 1/fps seconds. Each joint is
     clamped to ±limit around the reference: the current position for the
     first action, the previously sent goal for the following ones.

Run:
    .venv/bin/python -u scripts/run_policy_chunked.py \\
        --checkpoint checkpoints/cheese-v1/final \\
        --task "Spread the cheese relatively evenly over the pizza." \\
        --bus-serial 5B3E090065 --motor-ids 1,2,3,4,5,6,7,8 \\
        --gripper-motor-id 8 --actions-per-predict 5

Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch

from run_policy import (
    IMAGE_KEYS,
    SmolVLAPolicy,
    build_sync_write_command,
    fetch_fresh_frame,
    frame_to_batch,
    load_stats,
    new_station_client,
    send_commands,
    unnormalize_action,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--motor-ids", default="1,2,3,4,5,6",
                   help="Comma-separated motor IDs, same order as joints in the Frame.")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--actions-per-predict", type=int, default=5,
                   help="How many actions of each predicted chunk to execute before predicting again.")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Rate at which chunk actions are sent (dataset is ~10 Hz).")
    p.add_argument("--max-delta-ticks", type=int, default=200,
                   help="Per-action clamp for arm joints, in ticks. 0 to disable.")
    p.add_argument("--gripper-motor-id", type=int, default=0,
                   help="Motor ID of the gripper (gets --gripper-max-delta-ticks). 0 = no gripper.")
    p.add_argument("--gripper-max-delta-ticks", type=int, default=1700)
    p.add_argument("--max-chunks", type=int, default=0, help="Stop after N chunks. 0 = unlimited.")
    return p.parse_args()


@torch.no_grad()
def predict_chunk_raws(frame, policy, stats, task, device) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (raw goal ticks of shape (chunk_size, n_joints), per-joint ranges)."""
    batch, ranges = frame_to_batch(frame, stats, IMAGE_KEYS, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        pred_norm = policy.predict_action_chunk(batch)[0]
    goal = unnormalize_action(pred_norm.float(), stats).cpu().clamp(0.0, 1.0).numpy()

    rmin = np.array([r[0] for r in ranges], dtype=np.float64)
    rmax = np.array([r[1] for r in ranges], dtype=np.float64)
    raws = np.clip(np.round(rmin + goal * (rmax - rmin)), rmin, rmax).astype(np.int64)
    return raws, ranges


def clamp_to(reference: np.ndarray, target: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Move each joint from `reference` toward `target` by at most its limit (0 = no limit)."""
    lo = np.where(limits > 0, reference - limits, target)
    hi = np.where(limits > 0, reference + limits, target)
    return np.clip(target, lo, hi)


async def run(args) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    limits = np.array([args.gripper_max_delta_ticks if m == args.gripper_motor_id else args.max_delta_ticks
                       for m in motor_ids], dtype=np.int64)
    period = 1.0 / args.fps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = {k: v.to(device) for k, v in load_stats(args.checkpoint / "stats.safetensors").items()}
    print(f"loading {args.checkpoint} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config_overrides={"load_vlm_weights": False}, strict=False,
    ).to(device)
    policy.eval()

    print(f"task:      {args.task!r}")
    print(f"bus:       {args.bus_serial}  motors: {motor_ids}")
    print(f"chunking:  {args.actions_per_predict} actions per predict @ {args.fps:g} Hz")
    print(f"limits:    {limits.tolist()} ticks per action")
    print("\nauto: Ctrl-C to stop.\n")

    client = await new_station_client(args.server, logging.getLogger("run_policy_chunked"))
    last_id = b""
    chunk = 0
    t_run = time.perf_counter()
    while not (args.max_chunks and chunk >= args.max_chunks):
        chunk += 1
        t0 = time.perf_counter()
        frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)
        pos = np.array([int(j.get_position()) for j in frame.get_joints()], dtype=np.int64)
        raws, _ = predict_chunk_raws(frame, policy, stats, args.task, device)
        t_pred = time.perf_counter()

        n = min(args.actions_per_predict, len(raws))
        reference = pos
        n_clamped = 0
        next_send = time.perf_counter()
        for k in range(n):
            goal = clamp_to(reference, raws[k], limits)
            n_clamped += int((goal != raws[k]).any())
            await send_commands(client, [build_sync_write_command(args.bus_serial, motor_ids, goal.tolist())])
            reference = goal
            next_send += period
            await asyncio.sleep(max(0.0, next_send - time.perf_counter()))

        print(f"[chunk {chunk}] t={time.perf_counter() - t_run:6.1f}s  "
              f"predict {1000 * (t_pred - t0):.0f}ms  sent {n}  clamped {n_clamped}\n"
              f"  pos   {pos.tolist()}\n"
              f"  goal0 {raws[0].tolist()}\n"
              f"  goalN {raws[n - 1].tolist()}")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted — stopping.")


if __name__ == "__main__":
    main()
