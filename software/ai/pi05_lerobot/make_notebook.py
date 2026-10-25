"""Builds train_pi05_cheese.ipynb (run once locally; the notebook is what goes to Jupyter)."""
import json

LEROBOT_COMMIT = "d3793b7e36522321c1c455f0dc4826db3793dceb"  # same lerobot as the laptop

cells = [
("md", """# π₀.₅ fine-tune on the cheese dataset (LeRobot `pi05`)

- Dataset: `ragmishra108/cheese_v1_lerobot` (private, LeRobot v3, 10 episodes, 8 joints, 2 cams 224×224 @ 10 fps)
- Base model: `lerobot/pi05_base`
- Output: checkpoints pushed to `ragmishra108/pi05_cheese_v1` (private)

**Before running:** the HF account of your token must have accepted the license of
[google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) (gated, manual approval).

Run the cells top to bottom. Training runs in its own `uv` Python 3.12 venv (LeRobot needs ≥ 3.12), so the kernel's Python version doesn't matter."""),
("code", """# 0. Helper: run a shell command and stream its output (IPython `!` can't do multi-line commands)
import os, subprocess, sys

def sh(cmd, log=None):
    print("$", cmd if len(cmd) < 300 else cmd[:300] + " ...", flush=True)
    f = open(log, "a") if log else None
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    for line in p.stdout:
        print(line, end="", flush=True)
        if f: f.write(line)
    p.wait()
    if f: f.close()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}")

def py(code):
    \"\"\"Run Python code inside the lerobot venv.\"\"\"
    path = f"{WORK}/_snippet.py"
    open(path, "w").write(code)
    sh(f"{VENV}/bin/python {path}")

WORK = os.path.abspath("pi05_work"); os.makedirs(WORK, exist_ok=True)
VENV = f"{WORK}/.venv"
OUT = f"{WORK}/outputs/pi05_cheese_v1"
os.environ["HF_LEROBOT_HOME"] = f"{WORK}/lerobot_home"
print("work dir:", WORK)"""),
("code", """# 1. GPU check → picks a training profile
q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, check=True).stdout.strip().splitlines()
name, mem_mib, driver = [x.strip() for x in q[0].split(",")]
mem_gb = int(mem_mib) / 1024
print(f"{len(q)} GPU(s): {name}, {mem_gb:.0f} GB, driver {driver}")

# Estimates, not measurements: on CUDA OOM, halve BATCH here and rerun cell 5.
if mem_gb >= 70:
    PROFILE = dict(train_expert_only="false", freeze_vision_encoder="false", BATCH=32)   # full fine-tune
elif mem_gb >= 40:
    PROFILE = dict(train_expert_only="true", freeze_vision_encoder="true", BATCH=32)     # VLM frozen
elif mem_gb >= 22:
    PROFILE = dict(train_expert_only="true", freeze_vision_encoder="true", BATCH=8)
else:
    raise SystemExit(f"{mem_gb:.0f} GB is too little for pi05 training (need >= 24 GB)")
STEPS, SAVE_FREQ = 10_000, 2_500
print("profile:", PROFILE, "| steps:", STEPS, "| save every:", SAVE_FREQ)"""),
("code", f"""# 2. Environment: uv + Python 3.12 venv with lerobot[pi,training] (same commit as the laptop). ~5-10 min.
sh(f"{{sys.executable}} -m pip -q install uv")
sh(f"{{sys.executable}} -m uv venv --python 3.12 {{VENV}}")
sh(f'VIRTUAL_ENV={{VENV}} {{sys.executable}} -m uv pip install "lerobot[pi,training] @ git+https://github.com/huggingface/lerobot.git@{LEROBOT_COMMIT}"')
py("import torch, lerobot; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())")"""),
("code", """# 3. Hugging Face login (token needs WRITE access). Uses $HF_TOKEN if set, otherwise asks.
from getpass import getpass
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or getpass("HF token: ")
py(\"\"\"
from huggingface_hub import HfApi, hf_hub_download
print('logged in as', HfApi().whoami()['name'])
hf_hub_download('google/paligemma-3b-pt-224', 'tokenizer_config.json')
print('paligemma access OK')
\"\"\")"""),
("code", """# 4. Dataset check (downloads ~75 MB)
py(\"\"\"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('ragmishra108/cheese_v1_lerobot')
print(ds.num_episodes, 'episodes,', ds.num_frames, 'frames,', ds.fps, 'fps |', ds[0]['task'])
\"\"\")"""),
("code", """# 5. Train. Log also goes to pi05_work/train.log. First run downloads pi05_base (~14 GB).
sh(f\"\"\"{VENV}/bin/lerobot-train \\
  --dataset.repo_id=ragmishra108/cheese_v1_lerobot \\
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \\
  --policy.n_action_steps=10 \\
  --policy.train_expert_only={PROFILE['train_expert_only']} \\
  --policy.freeze_vision_encoder={PROFILE['freeze_vision_encoder']} \\
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda \\
  --policy.push_to_hub=false \\
  --batch_size={PROFILE['BATCH']} --num_workers=8 \\
  --steps={STEPS} --save_freq={SAVE_FREQ} --log_freq=50 \\
  --output_dir={OUT} --job_name=pi05_cheese_v1 --wandb.enable=false --seed=1000\"\"\",
   log=f"{WORK}/train.log")"""),
("code", """# 6. Push checkpoints to the Hub (private). Safe to re-run, also after an interrupted run.
#    Every saved step goes to step-NNNNNN/, the latest also to the repo root.
py(f\"\"\"
import os, glob
from huggingface_hub import HfApi
a = HfApi(); repo = 'ragmishra108/pi05_cheese_v1'
a.create_repo(repo, private=True, exist_ok=True)
for ck in sorted(glob.glob('{OUT}/checkpoints/[0-9]*/pretrained_model')):
    step = ck.split('/')[-2]
    a.upload_folder(repo_id=repo, folder_path=ck, path_in_repo='step-' + step)
    print('uploaded step', step)
a.upload_folder(repo_id=repo, folder_path=os.path.realpath('{OUT}/checkpoints/last/pretrained_model'), path_in_repo='.')
print('latest checkpoint at repo root ->', repo)
\"\"\")"""),
("code", """# 7. (only if the run was interrupted) Resume from the last checkpoint, then run cell 6 again
sh(f"{VENV}/bin/lerobot-train --config_path={OUT}/checkpoints/last/pretrained_model/train_config.json --resume=true",
   log=f"{WORK}/train.log")"""),
]

nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, (kind, src) in enumerate(cells):
    c = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i}", "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
json.dump(nb, open("train_pi05_cheese.ipynb", "w"), indent=1)
print("wrote train_pi05_cheese.ipynb with", len(nb["cells"]), "cells")
