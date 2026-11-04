#!/usr/bin/env bash
# Evaluate checkpoints with the authors' think35 protocol, one shard per GPU.
#   scripts/eval_all.sh "base p50c p25c p10c" math500,gsm8k
# Results: $WS_ROOT/eval/<model>/parts/s<g>of<N>/scores.json; combine with ws/merge.py $WS_ROOT/eval
set -euo pipefail
MODELS=${1:-"base p50c p25c p10c"}
SETS=${2:-math500,gsm8k}
export WS_ROOT=${WS_ROOT:-/data/billhowe/weightspace}
export HF_HOME=$WS_ROOT/hf
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$WS_ROOT/venv/bin/python
export PATH=$WS_ROOT/venv/bin:$PATH   # ninja etc. for vLLM kernel builds
mkdir -p "$WS_ROOT/logs"
for g in $(seq 0 $((NGPU - 1))); do
  (
    for m in $MODELS; do
      out=$WS_ROOT/eval/$m/parts/s${g}of${NGPU}
      CUDA_VISIBLE_DEVICES=$g $PY "$REPO/ws/evaluate.py" run "$WS_ROOT/models/$m" "$out" \
        --sets "$SETS" --protocol think35 --shard "$g/$NGPU" >> "$WS_ROOT/logs/${m}_s${g}.log" 2>&1
    done
  ) &
done
wait
$PY "$REPO/ws/merge.py" "$WS_ROOT/eval" $MODELS
