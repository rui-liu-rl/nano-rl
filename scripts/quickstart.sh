#!/usr/bin/env bash
# Quickstart: validate the whole library end-to-end on a Mac (CPU/MPS), no GPU.
#
# What it does, in ~2-5 minutes on a 16GB Mac:
#   1. trains a reward model on synthetic preference pairs (accuracy -> ~1.0)
#   2. trains DPO on the same pairs (preference accuracy -> ~1.0, margin grows)
#   3. plots both runs to PNGs and prints a first->last summary
#
# This exercises the training system + all the loss machinery and *shows you a
# curve* proving post-training actually moved the model. No code reading required.
#
# Uses a tiny 135M model so it stays well under your RAM. Sampling-based RL
# (GRPO/PPO) is heavier and lives on the GPU path — see scripts/vastai/.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
STEPS="${STEPS:-40}"
BS="${BS:-8}"
# Keep CPU threads modest so the machine stays responsive.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

PY="${PY:-python}"
echo ">>> using model=$MODEL steps=$STEPS  (override with MODEL=/STEPS= env vars)"

echo ">>> [1/3] reward model"
$PY examples/train_reward_model.py --model "$MODEL" --steps "$STEPS" \
    --batch-size "$BS" --out outputs/quickstart_rm

echo ">>> [2/3] DPO"
$PY examples/train_dpo.py --model "$MODEL" --steps "$STEPS" \
    --batch-size "$BS" --out outputs/quickstart_dpo

echo ">>> [3/3] plotting"
$PY examples/plot.py outputs/quickstart_rm/metrics.jsonl
$PY examples/plot.py outputs/quickstart_dpo/metrics.jsonl

echo
echo ">>> DONE. Open the curves:"
echo "      outputs/quickstart_rm/curve.png   (reward-model accuracy should climb to ~1.0)"
echo "      outputs/quickstart_dpo/curve.png   (DPO acc/margin should climb, loss should fall)"
