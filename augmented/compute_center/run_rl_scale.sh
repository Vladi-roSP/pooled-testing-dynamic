#!/usr/bin/env bash
# Train PPO bucket policies at scale (N=50) for the augmented DAPTS env.
# Intended for a compute center, NOT a laptop. See README.md.
#
# Grid: B in {2,3,4}, G in {3,5}, seeds {0,1,2}, 200k timesteps each.
# Each run trains and evaluates vs. augmented myopic greedy, saving a model and
# an eval CSV under results/cluster/rl/.
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="results/cluster/rl"
mkdir -p "$OUT"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "git commit: $COMMIT"

TIMESTEPS="${TIMESTEPS:-200000}"
N="${N:-50}"

for B in 2 3 4; do
  for G in 3 5; do
    for SEED in 0 1 2; do
      tag="N${N}_B${B}_G${G}_s${SEED}_${COMMIT}"
      echo "=== bucket train ${tag} (${TIMESTEPS} steps) ==="
      python -m augmented.rl_train bucket \
        --source random --N "$N" --B "$B" --G "$G" \
        --timesteps "$TIMESTEPS" --seed "$SEED" \
        --eval-episodes 1000 \
        --model-dir "$OUT" \
        2>&1 | tee "$OUT/${tag}.log"
    done
  done
done
echo "done. models + logs under $OUT"
