"""Export + check every episode as soon as it is recorded (run next to `record_episodes.py record`).

Polls the session JSON. For each newly saved episode it runs dataset-generator on
that episode's range, then smolvla_py/scripts/check_episodes.py on the parquet,
and prints one verdict line: OK / WARN / FAIL / DISCARDED (with the reason).

Run in a second terminal (from the norma-core root):
    python3 software/station/examples/record-episodes-py/watch_session.py \
        --session sessions/rc2.json --output ~/datasets/dataset-rc2 --min-grasps 4

Files land at <output>-epNNN.parquet, the same names `record_episodes.py generate` uses,
so the merge command afterwards is unchanged. Ctrl-C to stop.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from record_episodes import ANSI_RE, find_generator

ROOT = Path(__file__).resolve().parents[4]
CHECKER = ROOT / "software" / "ai" / "smolvla_py" / "scripts" / "check_episodes.py"
CHECKER_PY = ROOT / "software" / "ai" / "smolvla_py" / ".venv" / "bin" / "python"
DISCARD_RE = re.compile(r"Episode #\d+ discarded[^:]*: (.+)")  # not the "Episodes discarded: N" summary

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
COLORS = {"OK": GREEN, "WARN": YELLOW, "FAIL": RED, "DISCARDED": RED}


def load_session(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # not written yet, or caught mid-write


def export_episode(ep: dict, session: dict, args) -> tuple[Path | None, str]:
    out = Path(f"{args.output}-ep{ep['index']:03d}")
    cmd = [args.generator, "-no-tui",
           "-robot", session["robot"],
           "-from", str(ep["from"]), "-to", str(ep["to"]),
           "-episode.duration", str(int(ep["duration_s"]) + 60),  # never split a tagged episode
           "-episode.min-commands", str(args.min_commands),
           "-output", str(out), "-task", session["task"]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    log = ANSI_RE.sub("", res.stdout + res.stderr)
    parquet = out.with_name(out.name + ".parquet")
    if parquet.exists():
        return parquet, ""
    reason = DISCARD_RE.search(log)
    if reason:
        return None, reason.group(1).strip()
    return None, f"no frames exported (exit {res.returncode}); Station data for this range may already be overwritten"


def check(parquet: Path, task: str, args) -> tuple[str, str]:
    cmd = [str(CHECKER_PY), str(CHECKER), str(parquet), "--task", task, "--min-grasps", str(args.min_grasps),
           "--drops", str(args.drops)] + (["--home", args.home] if args.home else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():  # the one episode row of the table
        if line.startswith(parquet.name[-34:]):
            parts = line.split(None, 8)  # name ep status frames dur grasps gap end notes...
            status = parts[2]
            details = f"{parts[3]} frames, {parts[4]}, {parts[5]} grasps, end {parts[7]}"
            notes = parts[8] if len(parts) > 8 else ""
            return status, details + (f" | {notes}" if notes else "")
    return "FAIL", f"checker error: {(res.stderr or res.stdout).strip()[-200:]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", required=True, help="output prefix, e.g. ~/datasets/dataset-rc2")
    p.add_argument("--min-grasps", type=int, default=4)
    p.add_argument("--min-commands", type=int, default=100)
    p.add_argument("--drops", type=int, default=0, help="expected drops per episode (FAIL if different), 0 = off")
    p.add_argument("--home", help="home pose motors 1-7, comma list (passed to the checker)")
    p.add_argument("--generator", default="./dataset-generator")
    p.add_argument("--poll-s", type=float, default=1.0)
    args = p.parse_args()
    args.output = str(Path(args.output).expanduser())
    args.generator = find_generator(args.generator)

    print(f"watching {args.session} → {args.output}-epNNN.parquet  (Ctrl-C to stop)\n", flush=True)
    done: set[int] = set()
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "DISCARDED": 0}
    try:
        while True:
            session = load_session(args.session)
            for ep in (session or {}).get("episodes", []):
                if ep["index"] in done:
                    continue
                done.add(ep["index"])
                parquet, reason = export_episode(ep, session, args)
                if parquet:
                    status, info = check(parquet, session["task"], args)
                else:
                    status, info = "DISCARDED", reason + "  → re-record this episode"
                counts[status] += 1
                print(f"{time.strftime('%H:%M:%S')}  ep{ep['index']:03d} ({ep['duration_s']:.0f}s)  "
                      f"{COLORS[status]}{status:<9}{RESET} {info}", flush=True)
                print("           totals: " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
