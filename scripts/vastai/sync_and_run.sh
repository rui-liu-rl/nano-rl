#!/usr/bin/env bash
# Push this repo to the vast.ai instance, install deps, train on GPU, pull results
# back. Run this LOCALLY after launch.sh.
#
# Usage:
#   INSTANCE=<id> scripts/vastai/sync_and_run.sh                 # default: GRPO
#   INSTANCE=<id> ALGO=ppo scripts/vastai/sync_and_run.sh
#   INSTANCE=<id> STEPS=300 scripts/vastai/sync_and_run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

INSTANCE="${INSTANCE:-$(cat .vast_instance_id 2>/dev/null || true)}"
[ -z "$INSTANCE" ] && { echo "set INSTANCE=<id> (or run launch.sh first)"; exit 1; }
ALGO="${ALGO:-grpo}"
STEPS="${STEPS:-300}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

# Parse host/port from the vast ssh-url (ssh://root@HOST:PORT).
URL=$(vastai ssh-url "$INSTANCE")
HOST=$(echo "$URL" | sed -E 's#ssh://[^@]+@([^:]+):.*#\1#')
PORT=$(echo "$URL" | sed -E 's#.*:([0-9]+)$#\1#')
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
echo ">>> target: root@$HOST:$PORT  algo=$ALGO steps=$STEPS"

echo ">>> syncing repo -> instance"
rsync -az --delete -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    --exclude '.venv' --exclude 'outputs' --exclude '.git' --exclude '__pycache__' \
    ./ "root@$HOST:/workspace/nano-rl-infra/"

echo ">>> bootstrapping + training on GPU"
$SSH "cd /workspace/nano-rl-infra && bash scripts/vastai/train_gpu.sh '$ALGO' '$STEPS' '$MODEL'"

echo ">>> pulling results back to ./outputs/"
mkdir -p outputs
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    "root@$HOST:/workspace/nano-rl-infra/outputs/" ./outputs/

echo
echo ">>> done. curves are in ./outputs/$ALGO/curve.png"
echo ">>> remember to destroy the box:  vastai destroy instance $INSTANCE"
