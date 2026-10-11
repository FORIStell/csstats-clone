#!/usr/bin/env bash
# Runs one round and pushes the results. Safe to call repeatedly; skips if a round is already running.
set -euo pipefail
cd "$(dirname "$0")"
exec 9>/tmp/imggen-round.lock
flock -n 9 || { echo "round already running"; exit 0; }
# Fresh container: install deps if missing.
python3 -c "import torch, diffusers, peft, transformers" 2>/dev/null || {
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q -r ../requirements.txt peft
}
python3 round.py "$@"
cd ..
N=$(python3 -c "import json;print(json.load(open('train/state.json'))['round'])")
git add -A rounds train/state.json ROUNDS.md; [ -d lora ] && git add -A lora
git commit -qm "Training round $N

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01We4UxhZEwWtugJdJjF8u7y"
for d in 0 2 4 8 16; do sleep $d; git pull -q --rebase origin main && git push -q origin main && break; done
echo "pushed round $N"
