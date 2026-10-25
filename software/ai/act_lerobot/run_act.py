"""Run a LeRobot ACT checkpoint (trained on a convert_to_lerobot.py dataset) on the station.

Loop at --fps (dataset is ~10 Hz):
  1. Fetch a fresh Frame from `inference/normvla`.
  2. observation.state = position_norm, cam0/cam1 = decoded JPEGs (same as the conversion).
  3. ACT select_action: predicts a chunk and executes it action by action
     (re-predicts every --n-action-steps). Output is goal_norm per joint.
  4. goal_norm → raw ticks via the frame's (range_min, range_max), clamped to ±limit
     around the current position, sent as one ST3215 sync-write.

Run with the lerobot environment:
    ~/hack26/lerobot/.venv/bin/python -u run_act.py \
        --checkpoint ~/hack26/lerobot/outputs/train/act_cheese_v1/checkpoints/last/pretrained_model \
        --task "Spread the cheese relatively evenly over the pizza." \
        --bus-serial 5B3E090065 --motor-ids 1,2,3,4,5,6,7,8 --gripper-motor-id 8

Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

from lerobot.common.control_utils import predict_action  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTPolicy  # noqa: E402

QUEUE_ID = "inference/normvla"
ST3215_TARGET_POS_REGISTER = 0x2A
CAMERAS = ("cam0", "cam1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="…/checkpoints/<step|last>/pretrained_model directory from lerobot-train")
    p.add_argument("--task", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus-serial", required=True)
    p.add_argument("--motor-ids", default="1,2,3,4,5,6")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--n-action-steps", type=int, default=10,
                   help="Actions executed per ACT prediction (ACT predicts chunk_size=100).")
    p.add_argument("--max-delta-ticks", type=int, default=200,
                   help="Per-step clamp for arm joints around the current position. 0 = off.")
    p.add_argument("--gripper-motor-id", type=int, default=0)
    p.add_argument("--gripper-max-delta-ticks", type=int, default=1700)
    p.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    return p.parse_args()


async def fetch_fresh_frame(client, last_id: bytes, timeout: float) -> tuple[normvla.FrameReader, bytes]:
    while True:
        qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        if entry is None:
            raise RuntimeError(f"{QUEUE_ID} closed without delivering a frame ({qr.err})")
        frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
        fid = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(0.01)


def build_sync_write_command(bus_serial: str, motor_ids: list[int], raw_goals: list[int]) -> commands.DriverCommand:
    sync = st3215.ST3215SyncWriteCommand(
        address=ST3215_TARGET_POS_REGISTER,
        motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=int(raw).to_bytes(2, "little"))
                for mid, raw in zip(motor_ids, raw_goals)],
    )
    return commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND,
        body=st3215.Command(target_bus_serial=bus_serial, sync_write=sync).encode(),
    )


def frame_to_observation(frame: normvla.FrameReader) -> dict[str, np.ndarray]:
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    obs = {"observation.state": np.array([j.get_position_norm() for j in joints], dtype=np.float32)}
    for k, cam in enumerate(CAMERAS):
        with Image.open(io.BytesIO(bytes(images[k].get_jpeg()))) as im:
            obs[f"observation.images.{cam}"] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return obs


async def run(args) -> None:
    logging.basicConfig(level=logging.WARNING)
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    limits = np.array([args.gripper_max_delta_ticks if m == args.gripper_motor_id else args.max_delta_ticks
                       for m in motor_ids], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.checkpoint} on {device} ...")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.config.n_action_steps = args.n_action_steps
    policy.to(device).eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))

    print(f"task:      {args.task!r}")
    print(f"bus:       {args.bus_serial}  motors: {motor_ids}")
    print(f"chunking:  {args.n_action_steps} actions per predict @ {args.fps:g} Hz (ACT chunk {policy.config.chunk_size})")
    print(f"limits:    {limits.tolist()} ticks per step\n\nauto: Ctrl-C to stop.\n")

    client = await new_station_client(args.server, logging.getLogger("run_act"))
    period = 1.0 / args.fps
    last_id = b""
    step = 0
    t_run = time.perf_counter()
    next_tick = time.perf_counter()
    while not (args.max_steps and step >= args.max_steps):
        step += 1
        frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)
        joints = frame.get_joints() or []
        pos = np.array([int(j.get_position()) for j in joints], dtype=np.int64)
        rmin = np.array([int(j.get_range_min()) for j in joints], dtype=np.float64)
        rmax = np.array([int(j.get_range_max()) for j in joints], dtype=np.float64)

        t0 = time.perf_counter()
        action = predict_action(frame_to_observation(frame), policy, device, preprocessor, postprocessor,
                                use_amp=False, task=args.task)
        t_pred = time.perf_counter() - t0
        goal_norm = np.clip(action.detach().cpu().numpy().reshape(-1), 0.0, 1.0)
        raw = np.clip(np.round(rmin + goal_norm * (rmax - rmin)), rmin, rmax).astype(np.int64)
        goal = np.clip(raw, np.where(limits > 0, pos - limits, raw), np.where(limits > 0, pos + limits, raw))

        await send_commands(client, [build_sync_write_command(args.bus_serial, motor_ids, goal.tolist())])
        clamped = int((goal != raw).any())
        print(f"[step {step}] t={time.perf_counter() - t_run:6.1f}s  predict {1000 * t_pred:.0f}ms  "
              f"clamped {clamped}\n  pos   {pos.tolist()}\n  goal  {goal.tolist()}")

        next_tick += period
        await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted — stopping.")


if __name__ == "__main__":
    main()
