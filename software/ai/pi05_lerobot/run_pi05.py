"""Run a LeRobot pi05 checkpoint (from train_pi05_cheese.ipynb) on the station.

Same loop, observation mapping and safety clamp as ../act_lerobot/run_act.py; only the policy
loading differs (policy class is read from the checkpoint's config.json, so ACT works too).

Download the checkpoint first (private repo, ~7 GB):
    ~/hack26/lerobot/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('ragmishra108/pi05_cheese_v1', local_dir='checkpoints/pi05_cheese_v1', \
        allow_patterns=['*.json', '*.safetensors'], ignore_patterns=['step-*/*'])"

Run with the lerobot environment:
    ~/hack26/lerobot/.venv/bin/python -u run_pi05.py \
        --checkpoint checkpoints/pi05_cheese_v1 \
        --task "Spread the cheese relatively evenly over the pizza." \
        --bus-serial 5B3E090065 --motor-ids 1,2,3,4,5,6,7,8 --gripper-motor-id 8
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "act_lerobot"))

import run_act  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402


class _AnyPolicy:
    """Stand-in for ACTPolicy in run_act: picks the class named in the checkpoint config."""

    @staticmethod
    def from_pretrained(path):
        cfg = PreTrainedConfig.from_pretrained(path)
        print(f"policy type: {cfg.type}")
        return get_policy_class(cfg.type).from_pretrained(path)


def main() -> None:
    run_act.ACTPolicy = _AnyPolicy
    try:
        asyncio.run(run_act.run(run_act.parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted — stopping.")


if __name__ == "__main__":
    main()
