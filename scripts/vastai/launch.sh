#!/usr/bin/env bash
# Reserve a GPU box on vast.ai and print how to connect. Run this LOCALLY.
#
# Prereqs (one time):
#   pipx install vastai        # or: pip install --user vastai
#   vastai set api-key <KEY>   # from https://cloud.vast.ai/account/
#
# Usage:
#   scripts/vastai/launch.sh                 # cheapest single 24GB+ GPU
#   GPU="RTX_4090" scripts/vastai/launch.sh  # pick a GPU type
#   MAX_PRICE=0.5 scripts/vastai/launch.sh   # cap $/hr
#
# A single 24GB GPU (RTX 3090/4090) is plenty for Qwen2.5-0.5B GRPO/PPO.
set -euo pipefail

GPU="${GPU:-RTX_3090}"          # set to "" to allow any GPU
MAX_PRICE="${MAX_PRICE:-0.6}"   # $/hr ceiling
DISK="${DISK:-40}"              # GB
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"

command -v vastai >/dev/null || { echo "install the vast CLI: pipx install vastai"; exit 1; }

QUERY="reliability>0.98 num_gpus=1 disk_space>=${DISK} rentable=true dph<=${MAX_PRICE}"
[ -n "$GPU" ] && QUERY="$QUERY gpu_name=${GPU}"

echo ">>> searching offers: $QUERY"
OFFER=$(vastai search offers "$QUERY" -o 'dph+' --raw \
        | python3 -c 'import json,sys; o=json.load(sys.stdin); print(o[0]["id"]) if o else ""')
[ -z "$OFFER" ] && { echo "no offers matched. loosen GPU/MAX_PRICE."; exit 1; }
echo ">>> cheapest offer id: $OFFER"

echo ">>> creating instance (image=$IMAGE disk=${DISK}GB)"
vastai create instance "$OFFER" --image "$IMAGE" --disk "$DISK" \
    --ssh --direct --onstart-cmd "touch /root/onstart_done" \
    | tee /tmp/vast_create.txt

ID=$(grep -oE "'new_contract': [0-9]+" /tmp/vast_create.txt | grep -oE '[0-9]+' || true)
[ -z "$ID" ] && ID=$(vastai show instances --raw | python3 -c 'import json,sys; xs=json.load(sys.stdin); print(sorted(xs,key=lambda x:x["id"])[-1]["id"]) if xs else "")')
echo ">>> instance id: $ID  (save this)"
echo "$ID" > .vast_instance_id

echo ">>> waiting for it to come up (this can take a minute or two)..."
until vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["actual_status"])' 2>/dev/null | grep -q running; do
    sleep 5; printf '.'
done
echo

SSH_URL=$(vastai ssh-url "$ID")
echo
echo "=========================================================="
echo " instance $ID is up."
echo " ssh:  $SSH_URL"
echo
echo " next, push the code and run training in one shot:"
echo "   INSTANCE=$ID scripts/vastai/sync_and_run.sh"
echo
echo " when done, DESTROY it so you stop paying:"
echo "   vastai destroy instance $ID"
echo "=========================================================="
