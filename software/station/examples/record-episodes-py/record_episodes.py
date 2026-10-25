"""Terminal episode recorder for NormaCore Station.

Replaces the GUI steps "press Tag, copy frame number from History".

    record   SPACE starts an episode (tags <prefix>-epNNN-start), SPACE again ends it
             (tags <prefix>-epNNN-end). Frame pointers are saved to a session JSON.
    generate Runs dataset-generator once per episode from the saved session JSON.

Usage (from the norma-core root, so `target.gen_python` is importable):

    python3 software/station/examples/record-episodes-py/record_episodes.py record \
        --robot localhost:8888 --task "put the cube inside box"
    python3 software/station/examples/record-episodes-py/record_episodes.py generate \
        --session sessions/<file>.json --output ~/datasets/dataset-cube
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

# station_py lives in software/station/shared, protobufs in <norma-core>/target/gen_python
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "software" / "station" / "shared"))
sys.path.insert(0, str(ROOT))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

INFERENCE_STATES_QUEUE = "inference-states"
CT_ADD_TAG = 0
CT_REMOVE_TAG = 1

logger = logging.getLogger("record_episodes")


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_tag_command(cmd_type: int, ptr: bytes, tag: str) -> bytes:
    """Hand-encoded inference_tags.Command (the generated Python module is not built by default)."""
    tag_b = tag.encode()
    out = b""
    if cmd_type != 0:
        out += b"\x08" + _varint(cmd_type)
    out += b"\x12" + _varint(len(ptr)) + ptr
    out += b"\x1a" + _varint(len(tag_b)) + tag_b
    return out


async def send_tag(client, cmd_type: int, ptr: bytes, tag: str) -> None:
    cmd = commands.DriverCommand(
        type=drivers.StationCommandType.STC_INFERENCE_TAG_COMMAND,
        body=encode_tag_command(cmd_type, ptr, tag),
    )
    await send_commands(client, [cmd])


class Keys:
    """Single-keypress reader on a raw TTY."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def ptr_int(ptr: bytes) -> int:
    return int.from_bytes(ptr, "little")


async def record(args) -> None:
    client = await new_station_client(args.robot, logger)
    latest = {"ptr": None}

    async def follow_latest():
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        client.follow(INFERENCE_STATES_QUEUE, q)
        while True:
            entry = await q.get()
            latest["ptr"] = bytes(entry.ID.ID)

    follower = asyncio.create_task(follow_latest())
    while latest["ptr"] is None:
        await asyncio.sleep(0.05)

    prefix = args.prefix or time.strftime("%Y%m%d-%H%M%S")
    session_path = Path(args.sessions_dir) / f"{prefix}.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session = {"robot": args.robot, "task": args.task, "prefix": prefix, "episodes": []}
    current = None  # {"index", "tag", "from_ptr", "t0"}

    def save():
        session_path.write_text(json.dumps(session, indent=2))

    def status(msg: str) -> None:
        sys.stdout.write("\r\033[K" + msg)
        sys.stdout.flush()

    print(f"Session {prefix} -> {session_path}")
    print("SPACE start/stop episode | x discard current | q quit")
    with Keys() as keys:
        while True:
            await asyncio.sleep(0.02)
            if current:
                status(f"● ep{current['index']:03d} recording {time.time() - current['t0']:5.1f}s "
                       f"frame {ptr_int(latest['ptr'])}")
            else:
                status(f"○ idle, {len(session['episodes'])} saved, frame {ptr_int(latest['ptr'])}")

            key = keys.poll()
            if key is None:
                continue

            if key == " ":
                ptr = latest["ptr"]
                if current is None:
                    idx = len(session["episodes"]) + 1
                    tag = f"{prefix}-ep{idx:03d}-start"
                    await send_tag(client, CT_ADD_TAG, ptr, tag)
                    current = {"index": idx, "tag": tag, "from_ptr": ptr, "t0": time.time()}
                else:
                    tag = f"{prefix}-ep{current['index']:03d}-end"
                    await send_tag(client, CT_ADD_TAG, ptr, tag)
                    session["episodes"].append({
                        "index": current["index"],
                        "from": ptr_int(current["from_ptr"]),
                        "to": ptr_int(ptr),
                        "duration_s": round(time.time() - current["t0"], 1),
                    })
                    save()
                    print(f"\nep{current['index']:03d}: from={session['episodes'][-1]['from']} "
                          f"to={session['episodes'][-1]['to']}")
                    current = None
            elif key == "x" and current:
                await send_tag(client, CT_REMOVE_TAG, current["from_ptr"], current["tag"])
                print(f"\nep{current['index']:03d} discarded")
                current = None
            elif key == "q":
                if current:
                    print("\nstop the running episode first (SPACE) or discard it (x)")
                    continue
                break

    follower.cancel()
    save()
    print(f"\n{len(session['episodes'])} episodes saved to {session_path}")


def find_generator(path: str) -> str:
    for candidate in (path, shutil.which("dataset-generator"),
                      str(Path.home() / "Downloads" / "dataset-generator")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    sys.exit(f"dataset-generator not found at '{path}', on PATH or in ~/Downloads; pass --generator")


GAP_RE = re.compile(r"marked for skip at frame (\d+): frame gap (\d+)ms")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def generate(args) -> None:
    session = json.loads(Path(args.session).read_text())
    generator = find_generator(args.generator)
    # Episodes are already cut by our tags; dataset-generator only uses -episode.duration to split a range,
    # so keep it above the longest episode to never split one.
    max_dur = max(e["duration_s"] for e in session["episodes"])
    duration = args.episode_duration or int(max_dur) + 60

    def export(frm: int, to: int, out: str) -> tuple[str | None, tuple[int, int] | None]:
        """Run dataset-generator on [frm, to]. Return (parquet path or None, (gap_frame, gap_ms) or None)."""
        cmd = [generator, "-no-tui",
               "-robot", args.robot or session["robot"],
               "-from", str(frm), "-to", str(to),
               "-episode.duration", str(duration),
               "-episode.min-commands", str(args.min_commands),
               "-output", out, "-task", args.task or session["task"]]
        print("+", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log = res.stdout + res.stderr
        print(log, end="", flush=True)
        gap = GAP_RE.search(ANSI_RE.sub("", log))
        # dataset-generator exits 0 even when it discards the episode; it then writes no parquet
        path = out + ".parquet" if Path(out + ".parquet").exists() else None
        return path, (int(gap.group(1)), int(gap.group(2))) if gap else None

    out_files, discarded, split = [], [], []
    for ep in session["episodes"]:
        out = f"{args.output}-ep{ep['index']:03d}"
        path, gap = export(ep["from"], ep["to"], out)
        if path:
            out_files.append(path)
        elif gap and args.split_on_gap:
            # Station dropped frames once (e.g. a servo error): keep the clean parts before and after the gap.
            gap_frame, gap_ms = gap
            rate = (ep["to"] - ep["from"]) / ep["duration_s"]  # inference-states ids per second
            a_to = gap_frame - int((gap_ms / 1000 + 0.5) * rate)
            parts = [p for p in (export(ep["from"], a_to, out + "a")[0], export(gap_frame, ep["to"], out + "b")[0]) if p]
            out_files += parts
            split.append(f"ep{ep['index']:03d} → {len(parts)} part(s)")
            if not parts:
                discarded.append(ep["index"])
        else:
            discarded.append(ep["index"])
    if split:
        print("\nsplit at frame gaps: " + ", ".join(split))
    print(f"\n{len(out_files)} exported, {len(discarded)} discarded"
          + (f" (episodes {discarded}, see WRN lines above for the reason)" if discarded else ""))
    if not out_files:
        return
    print("\nMerge with:\n  uv run python scripts/merge_datasets.py --inputs "
          + " ".join(out_files) + f" --output {args.output}.parquet")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("record")
    r.add_argument("--robot", default="localhost:8888")
    r.add_argument("--task", required=True)
    r.add_argument("--prefix", help="tag prefix / session name (default: timestamp)")
    r.add_argument("--sessions-dir", default="sessions")

    g = sub.add_parser("generate")
    g.add_argument("--session", required=True)
    g.add_argument("--output", required=True, help="output prefix; episode N -> <output>-epNNN")
    g.add_argument("--robot", help="override robot address from session")
    g.add_argument("--task", help="override task from session")
    g.add_argument("--generator", default="./dataset-generator")
    g.add_argument("--episode-duration", type=int,
                   help="seconds; default: longest recorded episode plus 60 (never splits an episode)")
    g.add_argument("--min-commands", type=int, default=100)
    g.add_argument("--split-on-gap", action="store_true",
                   help="If an episode is discarded for a frame gap, export the parts before and after the gap.")

    args = p.parse_args()
    if args.mode == "record":
        asyncio.run(record(args))
    else:
        generate(args)


if __name__ == "__main__":
    main()
