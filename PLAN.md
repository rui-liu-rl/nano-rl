# nano-rl — Design Notes & Roadmap

> The README is the user-facing intro. This file is the design rationale and what's
> next — kept in the repo on purpose as part of building in public.

A minimal, from-scratch RL infrastructure for post-training small LLMs. Built for
learning: every piece is small enough to read in one sitting, with clean seams
between the three subsystems.

## Goals

- **Minimal & from-scratch.** No TRL / no vLLM / no DeepSpeed. Just PyTorch +
  HuggingFace `transformers` for the model definition and weights.
- **Three clean, modular subsystems** that can be developed and tested in isolation:
  1. **Inference engine** — generate rollouts from a policy (batched sampling loop
     with a KV cache, written by hand).
  2. **Training system** — forward/backward, log-prob computation, optimizer step,
     grad accumulation/clipping. Knows nothing about RL.
  3. **RL algorithms** — glue that orchestrates inference + training into a learning
     loop. PPO, GRPO, DPO, reward modeling.
- **Runs end-to-end on a small model** so you can iterate on a laptop/single GPU.

## Model choice

Default policy: **`Qwen/Qwen2.5-0.5B-Instruct`**.
- Tiny (0.5B), instruction-tuned, strong for its size, ubiquitous in RL research.
- GRPO/PPO actually move the needle on it for simple math tasks.
- Swappable via `--model`. Smoke tests use a randomly-initialized tiny config so the
  whole pipeline runs on CPU in seconds with no download.

## Task / reward (for the policy-gradient algos)

We need a *verifiable* reward to keep things honest and offline. Default task is a
**synthetic arithmetic task** ("What is 17 + 25? ... answer in \boxed{}") with a
deterministic correctness reward (extract the final number, compare to ground truth)
plus a small format reward. GSM8K is available as an optional loader.

This gives us, for free:
- prompts + ground-truth answers for **GRPO/PPO**,
- preference pairs (correct vs. wrong sample) for **DPO** and **reward modeling**.

## Architecture

```
nanorl/
  utils.py        device/seed/masked-ops/logging
  models.py       load policy / reference / value-head / reward model
  inference.py    InferenceEngine: hand-written batched sampling loop + KV cache
  trainer.py      Trainer: logprobs, entropy, values, backward, step
  data.py         arithmetic task + GSM8K + preference-pair builder
  rewards.py      correctness / format / length reward functions
  rl/
    grpo.py       group-relative PG, KL-to-ref, PPO-clip surrogate
    ppo.py        actor-critic, GAE, clipped policy + value loss
    dpo.py        direct preference optimization (pairwise)
    reward_model.py  Bradley-Terry reward model training
examples/
  train_grpo.py  train_ppo.py  train_dpo.py  train_reward_model.py
tests/
  test_smoke.py  CPU-only end-to-end sanity for every subsystem
```

### Key seams (so the three parts stay decoupled)

- **Inference → RL**: `InferenceEngine.generate(prompts, SamplingParams) -> [Rollout]`
  where a `Rollout` carries `prompt_ids`, `response_ids`, and sampling `logprobs`.
  The engine never sees rewards or losses.
- **Training ← RL**: `Trainer` exposes `logprobs_and_values(input_ids, ...)` and
  `optim_step(loss)`. It never samples and never computes advantages.
- **RL** owns reward computation, advantage estimation, and the loss. It calls the
  engine to roll out and the trainer to learn.

## Algorithms (what each file implements)

- **GRPO** — sample `G` completions/prompt, reward each, advantage = group
  z-score `(r - mean)/std`, broadcast to response tokens; PPO-clipped surrogate
  with per-token KL penalty to a frozen reference (no critic needed).
- **PPO** — policy with a scalar **value head**; per-token reward = terminal reward
  minus per-token KL; **GAE(λ)** advantages; clipped policy loss + clipped value
  loss + entropy bonus; multiple epochs over the rollout buffer.
- **DPO** — no sampling; pairwise `-logσ(β·((πθ-πref)_chosen - (πθ-πref)_rejected))`.
- **Reward modeling** — scalar-head model, Bradley-Terry
  `-logσ(r_chosen - r_rejected)`. Produces a reward model usable by PPO.

## Milestones

1. [x] Scaffold + plan + git.
2. [x] Core: `utils`, `models`, `inference`, `trainer`, `data`, `rewards`.
3. [x] RL: `grpo`, `ppo`, `dpo`, `reward_model`.
4. [x] Examples + CPU smoke tests; verified e2e on tiny random model.
5. [x] Quickstart that learns + plots (reward model / DPO on 135M, CPU).
6. [x] GPU path: one-command vast.ai launch/sync/run scripts.
7. [ ] Real GRPO/PPO run on Qwen2.5-0.5B-Instruct (arithmetic, then GSM8K).
8. [ ] LoRA support (cut memory so 0.5B sampling fits comfortably on a laptop).
9. [ ] Held-out eval harness + accuracy-over-time plots.
10. [ ] Optional W&B/TensorBoard logging behind the JSONL logger.

## Non-goals (for now)

Distributed/multi-GPU, paged attention, sequence packing, LoRA (easy add later),
fancy logging. Correctness and readability over speed.
