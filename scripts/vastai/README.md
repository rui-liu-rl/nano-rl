# Running the GPU experiments on vast.ai

Sampling-based RL (GRPO, PPO) on a real model needs a GPU. These scripts make
reserving a box and running a job as close to one command as possible, so you're
not fiddling in the web console every time.

**TL;DR**
```bash
pipx install vastai && vastai set api-key <YOUR_KEY>   # one time
scripts/vastai/launch.sh                                # reserve cheapest 24GB GPU
INSTANCE=$(cat .vast_instance_id) scripts/vastai/sync_and_run.sh   # train GRPO + pull curve
vastai destroy instance $(cat .vast_instance_id)        # STOP PAYING
```
The GRPO curve lands in `./outputs/grpo/curve.png` on your laptop.

---

## One-time setup

1. Make an account at <https://cloud.vast.ai/> and add credit.
2. Get your API key from <https://cloud.vast.ai/account/>.
3. Install the CLI and register the key:
   ```bash
   pipx install vastai        # or: pip install --user vastai
   vastai set api-key <YOUR_KEY>
   ```
4. Make sure you have an SSH key (`~/.ssh/id_ed25519.pub`). Add it at
   <https://cloud.vast.ai/account/> → "SSH Keys", or run
   `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`.

## The three scripts

| Script | Runs where | Does |
|---|---|---|
| `launch.sh` | your laptop | finds the cheapest matching GPU, creates the instance, waits until it's up, writes the id to `.vast_instance_id` |
| `sync_and_run.sh` | your laptop | `rsync`s this repo to the box, runs `train_gpu.sh`, then pulls `outputs/` back |
| `train_gpu.sh` | the GPU box | installs deps with `uv`, runs the chosen algo, plots the curve |

## Typical session

```bash
# 1. reserve a machine (override GPU/price as needed)
GPU=RTX_4090 MAX_PRICE=0.5 scripts/vastai/launch.sh

# 2. train. ALGO ∈ {grpo, ppo, dpo, rm}; STEPS/MODEL optional
INSTANCE=$(cat .vast_instance_id) ALGO=grpo STEPS=300 scripts/vastai/sync_and_run.sh

# 3. look at ./outputs/grpo/curve.png  — frac_correct should trend up

# 4. ALWAYS destroy when done (billing is per second it exists)
vastai destroy instance $(cat .vast_instance_id)
```

Want to iterate by hand on the box instead? `launch.sh` prints the raw `ssh` URL;
SSH in, `cd /workspace/nano-rl-infra`, and run the `examples/train_*.py` scripts
directly.

## Picking hardware

- **Qwen2.5-0.5B GRPO/PPO**: any single 24GB GPU (RTX 3090/4090, A5000). Cheap,
  fast. This is the default.
- Want a bigger policy (1.5B–3B)? Bump to an A6000/A100 40–80GB and raise
  `MAX_PRICE`. The code is single-GPU; no multi-GPU sharding here.

## Cost hygiene

- vast.ai bills for as long as the instance **exists**, not just while training.
  Destroy it the moment you've pulled your results.
- `vastai show instances` lists everything you currently have running.
- Stuck/expensive? `vastai destroy instance <id>` (or destroy all from the web
  console). Set a low `MAX_PRICE` so a fat-fingered search can't grab an A100.

## Troubleshooting

- **`no offers matched`** — loosen the filter: `GPU="" MAX_PRICE=1.0 scripts/vastai/launch.sh`.
- **SSH refused right after create** — the box is still booting; `launch.sh` waits
  for `running`, but the SSH daemon can lag a few more seconds. Re-run
  `sync_and_run.sh`.
- **First run is slow** — it downloads the model into `/workspace/hf_cache`. That
  cache persists on the instance for subsequent runs.
