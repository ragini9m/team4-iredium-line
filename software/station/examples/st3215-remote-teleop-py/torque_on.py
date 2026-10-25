"""Enable follower torque in place (no movement), so the Station produces inference/normvla frames.

Stopping the teleop disables follower torque; with torque off on every bus the Station
emits no normvla frames and a policy/planner waits forever. This first sets each
follower motor's goal to its present position, then enables torque, so the arm holds
where it is instead of jumping to a stale goal.

Run (from this directory), with the teleop stopped:
    uv run python torque_on.py            # enable on follower 5B3E090065
    uv run python torque_on.py --off      # disable again
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from software.station.shared.station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402

from commands import send_motor_commands, set_torque  # noqa: E402
from mirror import MotorCommand  # noqa: E402
from state import parse_motor_state  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus", default="5B3E090065", help="follower bus serial")
    p.add_argument("--off", action="store_true")
    args = p.parse_args()

    client = await new_station_client(args.server, logging.getLogger("torque_on"))
    q: asyncio.Queue = asyncio.Queue()
    client.follow("st3215/inference", q)
    entry = await asyncio.wait_for(q.get(), 10)
    state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
    bus = next(b for b in state.get_buses() if b.get_bus().get_serial_number() == args.bus)
    motors = {m.get_id(): parse_motor_state(m) for m in bus.get_motors()}
    ids = sorted(motors)

    if args.off:
        await set_torque(client, args.bus, ids, enable=False)
        print(f"torque OFF on {args.bus} motors {ids}")
        return
    # Goal = present position first, so enabling torque does not move the arm.
    await send_motor_commands(client, args.bus,
                              [MotorCommand(motor_id=i, goal=motors[i].present_position) for i in ids])
    await asyncio.sleep(0.2)
    await set_torque(client, args.bus, ids, enable=True)
    print(f"torque ON on {args.bus} motors {ids}, holding at "
          f"{[motors[i].present_position for i in ids]}")


if __name__ == "__main__":
    asyncio.run(main())
