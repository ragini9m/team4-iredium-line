"""Recipe planner: break a written recipe into topping steps and run one policy skill per step.

    recipe text ──parse──▶ plan [cheese, salami×6, mushroom×6] ──select──▶ skill (checkpoint + prompt)
                                                                   └──execute──▶ chunked policy loop
                                                                                 until done / timeout

Skills (checkpoint, prompt, timeout per topping) live in planner_skills.yaml.
Parsing is rule-based: clauses are split on "then", ",", "and", "finally", ...;
each clause that names a topping becomes a step, in written order. The plan is
printed and must be confirmed with Enter before the robot moves.

A step is done once the arm has left home to work and has been back home for
`home_hold_s`, or the skill's timeout is hit. The planner never commands joints
itself (competition rule: anything that outputs joint commands must be a learned
policy): each skill must end back home on its own, as recorded in the demos.

Run (from smolvla_py):
    .venv/bin/python -u scripts/planner.py --recipe "Spread the cheese evenly, then add 6 salami, then 6 mushrooms"
    .venv/bin/python -u scripts/planner.py --plan cheese,salami:6,mushroom:6      # skip parsing
    .venv/bin/python -u scripts/planner.py --recipe "..." --dry-run               # only parse + check skills
    .venv/bin/python -u scripts/planner.py --recipe "..." --simulate              # real models/frames, nothing sent

Ctrl-C aborts at any time (servos keep their last goal).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from run_policy import (
    SmolVLAPolicy,
    build_sync_write_command,
    fetch_fresh_frame,
    load_stats,
    new_station_client,
    send_commands,
)
from run_policy_chunked import clamp_to, predict_chunk_raws

ROOT = Path(__file__).resolve().parent.parent
# Raw ticks, motor 8; must match make_stage_dataset.py so training stages and runtime switching agree.
GRIPPER_OPEN, GRIPPER_CLOSED = 1950, 1850
DROP_MIN_DISTANCE = 600
# A retry (grip missed, gripper reopened at the bowl) moves the arm ~200 ticks between close and
# open; a real drop carries the piece bowl -> pizza (>= ~470 in rc4). Only the latter counts.
DROP_MIN_MOVE = 400

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
CLAUSE_SPLIT = re.compile(
    r"\b(?:and then|then|after that|afterwards|followed by|next|finally|lastly|and)\b|[,;.\n]",
    re.IGNORECASE,
)
ORDER_WORDS = re.compile(r"\b(before|after(?! that)|first)\b", re.IGNORECASE)
NEGATION_WORDS = re.compile(r"\b(no|not|without|don't|skip)\b", re.IGNORECASE)


@dataclass
class Step:
    skill: str
    count: int | None
    source: str      # the recipe clause this step came from
    stage: int | None = None    # staged skills: stage k of `stages`
    stages: int | None = None
    done: str = "home"          # "home": back home and holding; "drop": gripper released away from home

    @property
    def label(self) -> str:
        if self.stage:
            return f"{self.skill} {self.stage}/{self.stages}"
        return f"{self.skill} ×{self.count}" if self.count else self.skill


# ---------------------------------------------------------------------------
# Planning

def parse_recipe(text: str, skills: dict) -> tuple[list[Step], list[str]]:
    """Return (steps in written order, warnings for the operator)."""
    steps, warnings = [], []
    for clause in (c.strip() for c in CLAUSE_SPLIT.split(text) if c and c.strip()):
        found = [name for name, s in skills.items()
                 if any(re.search(rf"\b{re.escape(syn)}(?:s|es)?\b", clause, re.IGNORECASE) for syn in s["synonyms"])]
        if not found:
            continue
        if len(found) > 1:
            warnings.append(f"clause names several toppings, using written order: {clause!r}")
        if NEGATION_WORDS.search(clause):
            warnings.append(f"negation in clause, check it is meant as a step: {clause!r}")
        count = None
        m = re.search(r"\b(\d+)\b", clause) or re.search(rf"\b({'|'.join(NUMBER_WORDS)})\b", clause, re.IGNORECASE)
        if m:
            count = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1).lower()]
        found.sort(key=lambda n: min(m.start() for syn in skills[n]["synonyms"]
                                     for m in re.finditer(rf"\b{re.escape(syn)}", clause, re.IGNORECASE)))
        for name in found:
            steps.append(Step(name, count, clause))
    if ORDER_WORDS.search(text):
        warnings.append("recipe uses before/after/first: steps are in WRITTEN order, check it matches the intended order")
    names = [s.skill for s in steps]
    for dup in {n for n in names if names.count(n) > 1}:
        warnings.append(f"topping {dup!r} appears more than once")
    return steps, warnings


def parse_plan_override(spec: str) -> list[Step]:
    """'cheese,salami:6,mushroom:6' → steps."""
    steps = []
    for item in spec.split(","):
        name, _, count = item.strip().partition(":")
        steps.append(Step(name.strip(), int(count) if count else None, item.strip()))
    return steps


def resolve(steps: list[Step], skills: dict) -> tuple[list[tuple[Step, dict, str]], list[str]]:
    """Attach skill config + final prompt to each step. Return (resolved, errors)."""
    resolved, errors = [], []
    for st in steps:
        sk = skills.get(st.skill)
        if sk is None:
            errors.append(f"unknown skill {st.skill!r} (known: {', '.join(skills)})")
            continue
        if "{count}" in sk.get("prompt", sk.get("stage_prompt", "")):
            st.count = st.count or sk.get("default_count")
            if st.count not in sk.get("trained_counts", [st.count]):
                errors.append(f"{st.skill}: count {st.count} requested, policy is only trained for {sk['trained_counts']}")
        elif st.count:
            st.count = None  # e.g. "spread the cheese" has no count
        if not (ROOT / sk["checkpoint"] / "model.safetensors").exists():
            errors.append(f"{st.skill}: checkpoint {sk['checkpoint']} not found")
        if sk.get("stages"):
            # Staged skill: one prompt per piece. stage_done "drop" (make_stage_dataset.py split of whole
            # demos): switch prompt when the drop is detected, last stage ends home. stage_done "home"
            # (one recorded episode per piece): every stage ends back home. stage_done "drop_home": every
            # stage ends as soon as the piece was placed and the arm is near home again (no hold time, so
            # the policy cannot start a second pick at the rest pose under the same prompt).
            n = sk["stages"]
            stage_done = sk.get("stage_done", "drop")
            for k in range(1, n + 1):
                if stage_done in ("home", "drop_home"):
                    done = stage_done
                else:
                    done = "home" if k == n else "drop"
                stage = Step(st.skill, st.count, st.source, stage=k, stages=n, done=done)
                resolved.append((stage, sk, sk["stage_prompt"].format(k=k, n=n, count=st.count)))
        else:
            resolved.append((st, sk, sk["prompt"].format(count=st.count)))
    return resolved, errors


# ---------------------------------------------------------------------------
# Execution

class Robot:
    def __init__(self, client, cfg: dict, simulate: bool):
        r, e = cfg["robot"], cfg["execution"]
        self.client, self.simulate = client, simulate
        self.bus, self.motor_ids = r["bus_serial"], r["motor_ids"]
        self.home = np.array(r["home_pose"], dtype=np.int64)
        self.limits = np.array([e["gripper_max_delta_ticks"] if m == r["gripper_motor_id"] else e["max_delta_ticks"]
                                for m in self.motor_ids], dtype=np.int64)
        self.last_id = b""

    async def frame(self, timeout: float = 5.0):
        while True:
            try:
                frame, self.last_id = await asyncio.wait_for(
                    fetch_fresh_frame(self.client, self.last_id, timeout), timeout=3.0)
                break
            except asyncio.TimeoutError:
                # The Station only builds inference/normvla frames while exactly one bus has torque on;
                # stopping the teleop turns follower torque off.
                print("  waiting for frames: no new inference/normvla frame for 3 s. Is follower torque on? "
                      "(teleop folder: uv run python torque_on.py)", flush=True)
        return frame, np.array([int(j.get_position()) for j in frame.get_joints()], dtype=np.int64)

    async def send(self, goal: np.ndarray) -> None:
        if not self.simulate:
            await send_commands(self.client, [build_sync_write_command(self.bus, self.motor_ids, goal.tolist())])

    def dist_home(self, pos: np.ndarray) -> int:
        return int(np.abs(pos[:7] - self.home[:7]).sum())



async def run_skill(robot: Robot, policy, stats, prompt: str, sk: dict, cfg: dict, device, sim_chunks: int,
                    done_mode: str = "home", timeout_s: float | None = None) -> str:
    e, d = cfg["execution"], cfg["done"]
    period = 1.0 / e["fps"]
    timeout_s = timeout_s or sk["timeout_s"]
    # timeout_s = timeout_s + 10.0
    t0 = time.perf_counter()
    left_home = False
    home_since = None
    gripper_state = None
    placed_at = None  # drop_home: time the piece was placed
    history: list[tuple[float, np.ndarray]] = []
    chunk = 0
    while True:
        chunk += 1
        now = time.perf_counter() - t0
        frame, pos = await robot.frame()
        dist = robot.dist_home(pos)
        history.append((now, pos))

        # --- done / failure checks ---
        if done_mode in ("drop", "drop_home"):
            # Same rule as make_stage_dataset.py: gripper closed (grasp), then opens away from home.
            if pos[7] < GRIPPER_CLOSED and gripper_state != "closed":
                gripper_state, close_pose = "closed", pos[:7].copy()
            elif pos[7] > GRIPPER_OPEN and gripper_state == "closed":
                gripper_state = "open"
                if dist > DROP_MIN_DISTANCE and np.abs(pos[:7] - close_pose).sum() > DROP_MIN_MOVE:
                    if done_mode == "drop":
                        return f"done (drop after {now:.0f}s)"
                    placed_at = now
            if done_mode == "drop_home" and placed_at is not None and dist < d.get("placed_home_distance", 600):
                return f"done (placed at {placed_at:.0f}s, back home after {now:.0f}s)"
        if dist > d["work_distance"]:
            left_home, home_since = True, None
        elif left_home and dist < d["home_distance"]:
            home_since = home_since if home_since is not None else now
            if now - home_since >= d["home_hold_s"]:
                return f"done (back home after {now:.0f}s)"
        else:
            home_since = None
        if left_home and d["still_window_s"]:
            win = np.array([p[:7] for t, p in history if t >= now - d["still_window_s"]])
            if now >= d["still_window_s"] and (win.max(0) - win.min(0)).max() < d["still_ticks"]:
                return f"done (still for {d['still_window_s']}s after {now:.0f}s)"
        if not left_home and now > d["start_timeout_s"]:
            return f"FAILED (arm did not leave home within {d['start_timeout_s']}s)"
        if now > timeout_s:
            return f"TIMEOUT after {timeout_s}s"
        if robot.simulate and chunk > sim_chunks:
            return f"simulated ({sim_chunks} chunks)"

        # --- one chunk of the policy ---
        raws, _ = predict_chunk_raws(frame, policy, stats, prompt, device)
        reference = pos
        next_send = time.perf_counter()
        for k in range(min(e["actions_per_predict"], len(raws))):
            reference = clamp_to(reference, raws[k], robot.limits)
            await robot.send(reference)
            next_send += period
            await asyncio.sleep(max(0.0, next_send - time.perf_counter()))
        print(f"  [{now:5.1f}s] chunk {chunk:3d}  dist_home {dist:5d}  gripper {pos[7]:4d}"
              f"  {'working' if left_home else 'starting'}")


async def execute(plan, cfg: dict, simulate: bool, sim_chunks: int, auto_continue: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[str, tuple] = {}
    for _, sk, _ in plan:  # load every checkpoint up front, so switching steps is instant
        if sk["checkpoint"] not in models:
            ck = ROOT / sk["checkpoint"]
            print(f"loading {ck} ...")
            policy = SmolVLAPolicy.from_pretrained(ck, config_overrides={"load_vlm_weights": False}, strict=False)
            stats = {k: v.to(device) for k, v in load_stats(ck / "stats.safetensors").items()}
            models[sk["checkpoint"]] = (policy.to(device).eval(), stats)

    client = await new_station_client(cfg["robot"]["server"], logging.getLogger("planner"))
    robot = Robot(client, cfg, simulate)
    _, pos = await robot.frame()
    if robot.dist_home(pos) > cfg["execution"]["home_start_tolerance"] and not simulate:
        raise SystemExit(f"arm is not at the home pose (distance {robot.dist_home(pos)} > "
                         f"{cfg['execution']['home_start_tolerance']}). Move it there with teleop, stop teleop, retry.")

    results = []
    t_all = time.perf_counter()
    for i, (st, sk, prompt) in enumerate(plan, 1):
        print(f"\n=== step {i}/{len(plan)}: {st.label}  →  {prompt!r}")
        policy, stats = models[sk["checkpoint"]]
        t = time.perf_counter()
        outcome = await run_skill(robot, policy, stats, prompt, sk, cfg, device, sim_chunks, done_mode=st.done,
                                  timeout_s=sk.get("stage_timeout_s") if st.stage else None)
        results.append((st.label, outcome, time.perf_counter() - t))
        print(f"=== step {i} {outcome}")
        if not outcome.startswith(("done", "simulated")):
            if auto_continue:  # --yes: no operator input during evaluation, move on to the next step
                print("step did not finish cleanly: continuing with the next step (--yes)")
                continue
            ans = input("step did not finish cleanly: [c]ontinue with next step / [a]bort? ").strip().lower()
            if ans != "c":
                break

    print(f"\nsummary ({time.perf_counter() - t_all:.0f}s total)")
    for label, outcome, dt in results:
        print(f"  {label:<14} {dt:6.0f}s  {outcome}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--recipe", help="Written recipe, e.g. 'cheese, then 6 salami, then 6 mushrooms'")
    src.add_argument("--plan", help="Explicit plan, skips parsing: 'cheese,salami:6,mushroom:6'")
    p.add_argument("--config", type=Path, default=ROOT / "planner_skills.yaml")
    p.add_argument("--dry-run", action="store_true", help="Only parse and validate the plan.")
    p.add_argument("--simulate", action="store_true", help="Load models, read frames, but never send commands.")
    p.add_argument("--simulate-chunks", type=int, default=3, help="Chunks per step in --simulate.")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation before executing.")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = yaml.safe_load(args.config.read_text())
    skills = cfg["skills"]
    if args.recipe:
        steps, warnings = parse_recipe(args.recipe, skills)
    else:
        steps, warnings = parse_plan_override(args.plan), []
    plan, errors = resolve(steps, skills)

    print("plan:")
    for i, (st, sk, prompt) in enumerate(plan, 1):
        print(f"  {i}. {st.label:<12} prompt {prompt!r}\n     from {st.source!r}  ckpt {sk['checkpoint']}")
    if not plan:
        errors.append("no topping found in the recipe")
    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR:   {e}")
    if errors and not args.simulate:
        raise SystemExit("plan has errors, not executing (use --plan to override the parse)")
    if args.dry_run:
        return
    if not args.yes and input("\nexecute this plan? [Enter = yes, anything else = abort] ").strip():
        return
    try:
        asyncio.run(execute([x for x in plan if (ROOT / x[1]["checkpoint"] / "model.safetensors").exists()],
                            cfg, args.simulate, args.simulate_chunks, auto_continue=args.yes))
    except KeyboardInterrupt:
        print("\naborted — servos hold their last goal.")


if __name__ == "__main__":
    main()
