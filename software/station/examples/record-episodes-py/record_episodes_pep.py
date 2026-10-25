"""Pepperoni/salami variant of record_episodes.py: one episode per piece, each with its own prompt.

Same as record_episodes.py, plus per-stage mode (default 6 spots): the status line shows the
next spot, each saved episode stores task "Put salami on spot k of 6." and the spot advances
1 → 6 → 1. Digit keys 1-6 (while idle) set the next spot; `x` does not advance it.
`generate` and watch_session_pep.py export every episode with its own task.
The cheese tools (record_episodes.py, watch_session.py) are unchanged.

Usage (from the norma-core root, so `target.gen_python` is importable):

    python3 software/station/examples/record-episodes-py/record_episodes_pep.py record \
        --task "Spread the salami relatively evenly over the pizza." --prefix rs5
    python3 software/station/examples/record-episodes-py/record_episodes_pep.py generate \
        --session sessions/rs5.json --output ~/datasets/dataset-rs5
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
    if args.stages and "{k}" not in args.stage_prompt:
        sys.exit(f"--stage-prompt needs a {{k}} placeholder for the spot number, got {args.stage_prompt!r} "
                 "(e.g. \"Put salami on spot {k} of {n}.\")")
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
    # Per-stage mode: one episode per piece, stage k gets its own prompt; cycles 1..n, digit keys override.
    next_stage = 1

    def stage_task(k: int) -> str:
        return args.stage_prompt.format(k=k, n=args.stages)

    def save():
        session_path.write_text(json.dumps(session, indent=2))

    def status(msg: str) -> None:
        sys.stdout.write("\r\033[K" + msg)
        sys.stdout.flush()

    print(f"Session {prefix} -> {session_path}")
    print("SPACE start/stop episode | x discard current | q quit"
          + (f" | 1-{args.stages} set next spot" if args.stages else ""))
    with Keys() as keys:
        while True:
            await asyncio.sleep(0.02)
            spot = f"  [spot {current['stage'] if current else next_stage}/{args.stages}]" if args.stages else ""
            if current:
                status(f"● ep{current['index']:03d} recording {time.time() - current['t0']:5.1f}s "
                       f"frame {ptr_int(latest['ptr'])}{spot}")
            else:
                status(f"○ idle, {len(session['episodes'])} saved, frame {ptr_int(latest['ptr'])}  next{spot}")

            key = keys.poll()
            if key is None:
                continue

            if key == " ":
                ptr = latest["ptr"]
                if current is None:
                    idx = len(session["episodes"]) + 1
                    tag = f"{prefix}-ep{idx:03d}-start"
                    await send_tag(client, CT_ADD_TAG, ptr, tag)
                    current = {"index": idx, "tag": tag, "from_ptr": ptr, "t0": time.time(), "stage": next_stage}
                else:
                    tag = f"{prefix}-ep{current['index']:03d}-end"
                    await send_tag(client, CT_ADD_TAG, ptr, tag)
                    ep = {
                        "index": current["index"],
                        "from": ptr_int(current["from_ptr"]),
                        "to": ptr_int(ptr),
                        "duration_s": round(time.time() - current["t0"], 1),
                    }
                    if args.stages:
                        ep["stage"] = current["stage"]
                        ep["task"] = stage_task(current["stage"])
                        next_stage = current["stage"] % args.stages + 1
                    session["episodes"].append(ep)
                    save()
                    print(f"\nep{current['index']:03d}: from={ep['from']} to={ep['to']}"
                          + (f"  task={ep['task']!r}" if args.stages else ""))
                    current = None
            elif args.stages and current is None and key.isdigit() and 1 <= int(key) <= args.stages:
                next_stage = int(key)
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

    def export(frm: int, to: int, out: str, task: str) -> tuple[str | None, tuple[int, int] | None]:
        """Run dataset-generator on [frm, to]. Return (parquet path or None, (gap_frame, gap_ms) or None)."""
        cmd = [generator, "-no-tui",
               "-robot", args.robot or session["robot"],
               "-from", str(frm), "-to", str(to),
               "-episode.duration", str(duration),
               "-episode.min-commands", str(args.min_commands),
               "-output", out, "-task", task]
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
        task = args.task or ep.get("task") or session["task"]  # per-stage sessions store a task per episode
        path, gap = export(ep["from"], ep["to"], out, task)
        if path:
            out_files.append(path)
        elif gap and args.split_on_gap:
            # Station dropped frames once (e.g. a servo error): keep the clean parts before and after the gap.
            gap_frame, gap_ms = gap
            rate = (ep["to"] - ep["from"]) / ep["duration_s"]  # inference-states ids per second
            a_to = gap_frame - int((gap_ms / 1000 + 0.5) * rate)
            parts = [p for p in (export(ep["from"], a_to, out + "a", task)[0],
                                 export(gap_frame, ep["to"], out + "b", task)[0]) if p]
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
    r.add_argument("--stages", type=int, default=6,
                   help="per-stage mode: one episode per piece; spot number cycles 1..N (digit keys override)")
    r.add_argument("--stage-prompt", default="Put salami on spot {k} of {n}.",
                   help="per-episode task in per-stage mode ({k} = spot, {n} = stages)")

    g = sub.add_parser("generate")
    g.add_argument("--session", required=True)
    g.add_argument("--output", required=True, help="output prefix; episode N -> <output>-epNNN")
    g.add_argument("--robot", help="override robot address from session")
    g.add_argument("--task", help="override task from session")
    g.add_argument("--generator", default="./dataset-generator")
    g.add_argument("--episode-duration", type=int,
                   help="seconds; default: longest recorded episode plus 60 (never splits an episode)")
    g.add_argument("--min-commands", type=int, default=50)  # one-slice episodes are short
    g.add_argument("--split-on-gap", action="store_true",
                   help="If an episode is discarded for a frame gap, export the parts before and after the gap.")

    args = p.parse_args()
    if args.mode == "record":
        asyncio.run(record(args))
    else:
        generate(args)


if __name__ == "__main__":
    main()
