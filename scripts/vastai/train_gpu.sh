#!/usr/bin/env bash
# Runs ON the vast.ai instance (invoked by sync_and_run.sh). Installs deps with uv
# and runs sampling-based RL on a real small model, then plots the curve.
set -euo pipefail
ALGO="${1:-grpo}"
STEPS="${2:-300}"
MODEL="${3:-Qwen/Qwen2.5-0.5B-Instruct}"
cd "$(dirname "$0")/../.."

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false

# uv: fast, reproducible installs. Falls back to pip if curl is unavailable.
if ! command -v uv >/dev/null; then
    echo ">>> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh || pip install uv
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">>> installing deps (torch usually preinstalled in the image)"
uv pip install --system -q transformers numpy matplotlib datasets || \
    pip install -q transformers numpy matplotlib datasets

python - <<'PY'
import torch
print(">>> torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

OUT="outputs/$ALGO"
echo ">>> training $ALGO for $STEPS steps on $MODEL"
case "$ALGO" in
  grpo) python examples/train_grpo.py --model "$MODEL" --steps "$STEPS" \
            --group-size 8 --prompts-per-step 16 --out "$OUT" ;;
  ppo)  python examples/train_ppo.py  --model "$MODEL" --steps "$STEPS" \
            --rollouts-per-prompt 4 --prompts-per-step 16 --out "$OUT" ;;
  dpo)  python examples/train_dpo.py  --model "$MODEL" --steps "$STEPS" --out "$OUT" ;;
  rm)   python examples/train_reward_model.py --model "$MODEL" --steps "$STEPS" --out "$OUT" ;;
  *) echo "unknown ALGO=$ALGO (use grpo|ppo|dpo|rm)"; exit 1 ;;
esac

python examples/plot.py "$OUT/metrics.jsonl"
echo ">>> curve at $OUT/curve.png"
