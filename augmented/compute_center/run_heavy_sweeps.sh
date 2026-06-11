#!/usr/bin/env bash
# Large-N optimal-vs-greedy sweeps for the augmented scheme.
# Intended for a compute center, NOT a laptop. See README.md.
#
# Needs a Gurobi or MOSEK license for the optimal large-N pool selection; without
# one, pool_solvers falls back to the heuristic pool (see the import fix), so the
# run still completes but the "optimal" columns become heuristic.
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="results/cluster/sweeps"
mkdir -p "$OUT"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "git commit: $COMMIT"

SOLVER="${SOLVER:-gurobi}"          # gurobi | mosek | enum
INSTANCES="${INSTANCES:-50}"

echo "=== overnight: large-N optimal vs greedy (solver=$SOLVER) ==="
python augmented/overnight_experiments.py \
  --n 20 30 50 --B 2 3 --G 5 \
  --instances "$INSTANCES" --solver "$SOLVER" \
  2>&1 | tee "$OUT/overnight_${COMMIT}.log"

echo "=== sprint3: main battery at full instance counts ==="
python -c "
import sys; sys.path.insert(0, '.')
from augmented.sprint3_experiments import run_main_experiments
run_main_experiments(n_instances=${INSTANCES}, output_dir='${OUT}')
" 2>&1 | tee "$OUT/sprint3_main_${COMMIT}.log"

echo "done. CSVs + logs under $OUT"
