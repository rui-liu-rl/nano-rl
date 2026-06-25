# nano-rl

**A minimal, from-scratch RL stack for post-training small LLMs — built to be read.**

No TRL, no vLLM, no DeepSpeed. Just PyTorch + HuggingFace `transformers` for the
model, and a few hundred lines of clean code for everything else: a hand-written
inference engine, a tiny training loop, and the popular RL algorithms wired on top
— **PPO, GRPO, DPO, and reward modeling**.

It's small enough to read end-to-end in an afternoon, and it actually learns: you
can validate the whole pipeline on a laptop in a couple of minutes and *see the
curve*.

```
prompts ──▶ [ inference engine ] ──▶ rollouts ──▶ [ RL algorithm ] ──▶ loss ──▶ [ trainer ] ──▶ updated policy
             hand-written sampler        (reward + advantage)            AdamW step
```

---

## Why this exists

Most RL-for-LLM codebases are production frameworks: fast, general, and nearly
impossible to learn from. `nano-rl` is the opposite — a teaching implementation
with three clean, decoupled subsystems you can understand in isolation:

| Subsystem | File | Responsibility | Knows nothing about |
|---|---|---|---|
| **Inference engine** | `nanorl/inference.py` | batched autoregressive sampling with a KV cache, by hand | rewards, losses |
| **Training system** | `nanorl/trainer.py` | log-probs / entropy / values, backward, optimizer step | sampling, advantages |
| **RL algorithms** | `nanorl/rl/*.py` | reward, advantage estimation, the loss | how tokens are generated or how grads are applied |

The seams are deliberate: the RL layer calls the engine to roll out and the trainer
to learn, and nothing leaks across.

---

## Quickstart — validate it works in ~3 minutes (CPU, no GPU)

You don't need to read any code first. This trains a reward model and a DPO policy
on a tiny 135M model, then plots curves so you can *see* post-training worked.

```bash
# 1. install (uv is the fast modern way; see "Setup" below for pip/conda)
uv venv && uv pip install -e ".[viz]"

# 2. run the end-to-end validation (downloads a ~135M model, CPU-only)
uv run scripts/quickstart.sh
```

Open the PNGs it prints. You should see the reward model's loss collapse to ~0 with
chosen/rejected rewards pulling apart, and DPO's preference margin shoot up:

| Reward model | DPO |
|---|---|
| ![reward model curve](assets/quickstart_reward_model.png) | ![dpo curve](assets/quickstart_dpo.png) |

```
reward model:  loss 0.48 -> 0.00   r_chosen -0.9 -> +8.2   r_rejected -1.5 -> -6.0
DPO:           loss 0.69 -> 0.0004  acc 0.50 -> 1.00         margin -11 -> +67
```

If those curves move, every layer of the stack — tokenization, the training loop,
the loss, the optimizer — is wired up correctly.

> The quickstart uses a deliberately *easy, learnable* preference (a well-formed
> answer vs. a non-answer) so the signal is unambiguous on a tiny model. The
> harder "correct vs. wrong number" preference, and the sampling-based RL algos,
> are where the GPU comes in.

---

## What runs where

`nano-rl` is single-GPU and unsharded by design. The split is about **sampling**:
preference methods (DPO, reward modeling) just do forward/backward and are happy on
a laptop; GRPO/PPO generate rollouts every step and want a GPU.

| Task | Algorithm | Mac (CPU/MPS, 16GB) | GPU (vast.ai) |
|---|---|:---:|:---:|
| Smoke tests (random tiny model) | all | ✅ seconds | ✅ |
| **Reward modeling** (135M) | Bradley-Terry | ✅ ~1 min | ✅ |
| **DPO** (135M) | DPO | ✅ ~1 min | ✅ |
| **GRPO** (Qwen2.5-0.5B) | GRPO | 🐢 works but slow | ✅ recommended |
| **PPO** (Qwen2.5-0.5B) | PPO + value head | 🐢 works but slow | ✅ recommended |

On a Mac, keep to the 135M model and the preference methods. Reaching for a 0.5B
policy with sampling? Use the GPU path — see [`scripts/vastai/`](scripts/vastai/),
which reduces "reserve a box, run, pull results, shut it down" to a couple of
commands.

```bash
# on the GPU box, one algo end-to-end:
scripts/vastai/launch.sh                                  # reserve cheapest 24GB GPU
INSTANCE=$(cat .vast_instance_id) ALGO=grpo scripts/vastai/sync_and_run.sh
vastai destroy instance $(cat .vast_instance_id)          # stop paying
```

---

## The algorithms

| Algo | File | One-liner | Needs |
|---|---|---|---|
| **GRPO** | `rl/grpo.py` | sample a group/prompt, advantage = group z-score, PPO-clip + KL-to-ref | policy + frozen ref |
| **PPO** | `rl/ppo.py` | actor-critic, GAE(λ), clipped policy + value loss, per-token KL reward | policy + value head + ref |
| **DPO** | `rl/dpo.py` | pairwise `-logσ(β·Δ(logπθ−logπref))`, no sampling, no reward model | policy + frozen ref |
| **Reward model** | `rl/reward_model.py` | scalar head, Bradley-Terry `-logσ(r⁺−r⁻)` | one model |

The default task is a fully-offline **synthetic arithmetic** problem with a
*verifiable* reward (extract the `\boxed{}` answer, compare to ground truth), so the
RL loop has trustworthy signal with no dataset download. A GSM8K loader is included
for the real thing.

---

## Setup

**uv (recommended — fast, reproducible):**
```bash
# install uv once:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv                          # creates .venv
uv pip install -e ".[viz]"       # core + plotting; add ,data for GSM8K
uv run python tests/test_smoke.py
```

**pip / conda:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz]"
python tests/test_smoke.py
```

The import package is `nanorl`. Smoke tests use a random-init tiny model and need no
download or GPU.

---

## Repository layout

```
nanorl/
  inference.py    InferenceEngine — hand-written batched sampler + KV cache
  trainer.py      Trainer — logprobs/entropy/values, backward, optim step
  models.py       load policy / frozen ref / value head / reward model
  data.py         arithmetic task, GSM8K loader, preference-pair builder
  rewards.py      verifiable reward functions
  utils.py        device/seed, masked ops, log-prob helpers
  rl/
    grpo.py  ppo.py  dpo.py  reward_model.py   common.py
examples/
  train_grpo.py  train_ppo.py  train_dpo.py  train_reward_model.py  plot.py
scripts/
  quickstart.sh           # CPU end-to-end validation + curves
  vastai/                 # one-command GPU runs on vast.ai
tests/
  test_smoke.py           # CPU end-to-end sanity for every subsystem
```

Every example takes `--model`, `--steps`, `--out`, and `--smoke` (random tiny model,
CPU). Metrics stream to `outputs/<algo>/metrics.jsonl`; `examples/plot.py` turns any
run into a PNG.

---

## Design notes & roadmap

See [`PLAN.md`](PLAN.md) for the design rationale and what's next (real Qwen runs,
LoRA, GSM8K eval, logging). Built in public — issues and ideas welcome.

## Acknowledgements

Standing on the shoulders of TRL, DeepSpeed-Chat, and the DeepSeekMath (GRPO),
PPO-for-RLHF, and DPO papers. This repo trades their speed and generality for
something you can actually read.
