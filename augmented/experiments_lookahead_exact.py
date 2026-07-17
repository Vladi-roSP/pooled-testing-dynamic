"""Re-medicion de la ley de recuperacion del lookahead, ambos cableados.

La tabla vieja (99/40/16, lineas_research_francisco.md §2) midio el lookahead
LEGACY (_lookahead_best_pool: PB + updates secuenciales). Este script corre,
sobre instancias IDENTICAS y sembradas, las dos politicas:

  - legacy : pool elegido por greedy._lookahead_best_pool (cableado viejo);
             su valor se evalua con pesos de rama EXACTOS (el numero
             reportado es el valor verdadero de esa politica).
  - exacta : lookahead_exact.exact_lookahead_expected_utility (seleccion y
             pesos sobre el conjunto de perfiles consistentes).

Por B se reporta: hueco miope, hueco lookahead y recuperacion, por cableado.
CSV por instancia en augmented/data/lookahead_law_rewired.csv.

Uso:  PYTHONPATH=. python3 augmented/experiments_lookahead_exact.py
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask
from augmented.bayesian import bayesian_update_single_test
from augmented.greedy import (greedy_myopic_expected_utility,
                              _lookahead_best_pool, _branch_pmf)
from augmented.lookahead_exact import exact_lookahead_expected_utility
from augmented.solver import solve_optimal_dapts

N = 6
G = 4
B_VALUES = (1, 2, 3, 4)
NUM_INSTANCES = 30
BASE_SEED = 20260716
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "lookahead_law_rewired.csv")


def legacy_lookahead_expected_utility(p, u, B, G):
    """Valor VERDADERO (pesos de rama exactos) de la politica lookahead con
    el cableado legacy: seleccion via _lookahead_best_pool (PB + updates
    secuenciales), re-planificada en cada paso."""
    n = len(p)
    prior = list(p)

    def recurse(current_p, history, b, cleared):
        if b == 0:
            return sum(u[i] for i in indices_from_mask(cleared, n))
        pool, _ = _lookahead_best_pool(current_p, u, G, n, b, cleared)
        if pool == 0:
            return sum(u[i] for i in indices_from_mask(cleared, n))
        pool_idx = indices_from_mask(pool, n)
        pmf = _branch_pmf(prior, history, current_p, pool, pool_idx, n)
        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(current_p, pool, r, n)
            new_cleared = cleared | pool if r == 0 else cleared
            ev += pmf[r] * recurse(new_p, history + ((pool, r),),
                                   b - 1, new_cleared)
        return ev

    return recurse(list(p), (), B, 0)


def draw_instance(seed):
    rng = random.Random(seed)
    p = [rng.uniform(0.05, 0.5) for _ in range(N)]
    u = [rng.uniform(1.0, 5.0) for _ in range(N)]
    return p, u


def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fieldnames = ["B", "instance", "seed", "opt", "myopic",
                  "lookahead_legacy", "lookahead_exact"]
    rows = []
    for B in B_VALUES:
        for inst in range(NUM_INSTANCES):
            seed = BASE_SEED + inst
            p, u = draw_instance(seed)
            opt, _ = solve_optimal_dapts(p, u, B, G)
            myo = greedy_myopic_expected_utility(p, u, B, G)
            leg = legacy_lookahead_expected_utility(p, u, B, G)
            exa = exact_lookahead_expected_utility(p, u, B, G)
            assert myo <= opt + 1e-9 and leg <= opt + 1e-9 \
                and exa <= opt + 1e-9, (B, inst, opt, myo, leg, exa)
            rows.append({"B": B, "instance": inst, "seed": seed,
                         "opt": opt, "myopic": myo,
                         "lookahead_legacy": leg, "lookahead_exact": exa})
        print(f"B={B}: {NUM_INSTANCES} instancias listas", flush=True)

    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {CSV_PATH}\n")

    header = (f"{'B':>2} | {'hueco miope':>11} | "
              f"{'hueco LA legacy':>15} | {'recupera legacy':>15} | "
              f"{'hueco LA exacto':>15} | {'recupera exacto':>15}")
    print(header)
    print("-" * len(header))
    for B in B_VALUES:
        sub = [r for r in rows if r["B"] == B]
        gap_m = sum((r["opt"] - r["myopic"]) / r["opt"] for r in sub) / len(sub)
        gap_l = sum((r["opt"] - r["lookahead_legacy"]) / r["opt"]
                    for r in sub) / len(sub)
        gap_e = sum((r["opt"] - r["lookahead_exact"]) / r["opt"]
                    for r in sub) / len(sub)
        rec_l = 1.0 - gap_l / gap_m if gap_m > 1e-12 else 1.0
        rec_e = 1.0 - gap_e / gap_m if gap_m > 1e-12 else 1.0
        print(f"{B:>2} | {100*gap_m:>10.2f}% | {100*gap_l:>14.2f}% | "
              f"{100*rec_l:>14.0f}% | {100*gap_e:>14.2f}% | "
              f"{100*rec_e:>14.0f}%")


if __name__ == "__main__":
    main()
