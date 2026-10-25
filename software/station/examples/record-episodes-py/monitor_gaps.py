"""Live monitor for recording problems: frame gaps in inference/normvla and ST3215 motor errors.

dataset-generator discards any episode with a frame gap > 500 ms. The Station drops
normvla frames while a motor on the bus reports an error (overload, overheat, ...),
so this prints both, with timestamps, to find which motor/movement causes the gaps.

Run in a second terminal while recording or teleoperating (from the norma-core root):
    python3 software/station/examples/record-episodes-py/monitor_gaps.py --bus 5B3E090065
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "software" / "station" / "shared"))
sys.path.insert(0, str(ROOT))

from station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215 as st3215_pb  # noqa: E402

STATUS_ADDR = 0x41          # RamRegister::Status
CURRENT_ADDR = 0x45         # present current, 2 bytes
ERROR_BITS = {0: "Voltage", 1: "AngleLimit", 2: "Overheat", 3: "Range", 4: "Checksum", 5: "Overload", 6: "Instruction"}


def stamp() -> str:
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}"


async def watch_frames(client, gap_ms: int) -> None:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    client.follow("inference/normvla", q)
    last = None
    while True:
        try:
            await asyncio.wait_for(q.get(), timeout=3.0)
        except asyncio.TimeoutError:
            print(f"{stamp()}  no normvla frames for 3 s (Station needs exactly one bus with torque on)", flush=True)
            continue
        now = time.monotonic()
        if last is not None and (now - last) * 1000 > gap_ms:
            print(f"{stamp()}  FRAME GAP {1000 * (now - last):.0f} ms  (episode would be discarded)", flush=True)
        last = now


async def watch_motors(client, bus: str) -> None:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    client.follow("st3215/inference", q)
    active: dict[int, str] = {}
    while True:
        entry = await q.get()
        state = st3215_pb.InferenceStateReader(memoryview(bytes(entry.Data)))
        for b in state.get_buses() or []:
            if b.get_bus().get_serial_number() != bus:
                continue
            for m in b.get_motors() or []:
                raw = bytes(m.get_state() or b"")
                status = raw[STATUS_ADDR] if len(raw) > STATUS_ADDR else 0
                errs = ",".join(n for bit, n in ERROR_BITS.items() if status & (1 << bit))
                mid = m.get_id()
                if errs and active.get(mid) != errs:
                    cur = int.from_bytes(raw[CURRENT_ADDR:CURRENT_ADDR + 2], "little") if len(raw) > CURRENT_ADDR + 1 else -1
                    print(f"{stamp()}  MOTOR {mid} ERROR {errs}  (status 0x{status:02X}, current {cur})", flush=True)
                    active[mid] = errs
                elif not errs and mid in active:
                    print(f"{stamp()}  motor {mid} error cleared", flush=True)
                    del active[mid]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="localhost:8888")
    p.add_argument("--bus", default="5B3E090065", help="bus serial to watch for motor errors (follower)")
    p.add_argument("--gap-ms", type=int, default=500)
    args = p.parse_args()
    client = await new_station_client(args.robot, logging.getLogger("monitor"))
    print(f"watching inference/normvla gaps > {args.gap_ms} ms and motor errors on bus {args.bus}. Ctrl-C to stop.",
          flush=True)
    await asyncio.gather(watch_frames(client, args.gap_ms), watch_motors(client, args.bus))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
