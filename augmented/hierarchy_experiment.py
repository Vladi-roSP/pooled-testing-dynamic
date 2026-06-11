"""
Utility-hierarchy experiment (corrected code, branch correctness-and-paper).

For each (N, B, G) config and K seeded random instances, computes the six
welfare quantities and verifies the chain

    U_single <= U_s_NO <= U_s_O <= U_D <= U_D_A <= U_max

and reports the means plus the AUGMENTED BENEFIT of the count test over the
binary (classical) test, U_D_A - U_D (absolute and % of U_D). This is the
paper's headline empirical result, recomputed after the correctness fixes.

Instances: p_i ~ U(0,1), u_i ~ Uniform{1,2,3} (matching the preprint setup).

Run:  PYTHONPATH=. python augmented/hierarchy_experiment.py [--instances K] [--quick]
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.baselines import u_max, u_single
from augmented.static_solver import (solve_static_non_overlapping,
                                     solve_static_overlapping)
from augmented.classical_solver import solve_classical_dynamic
from augmented.solver import solve_optimal_dapts


def _val(x):
    """Solvers return either a scalar or (value, extra); normalize to scalar."""
    return x[0] if isinstance(x, tuple) else x


def hierarchy_for_instance(p, u, B, G):
    u_s = _val(u_single(p, u, B))
    u_sno = _val(solve_static_non_overlapping(p, u, B, G))
    u_so = _val(solve_static_overlapping(p, u, B, G))
    u_d = _val(solve_classical_dynamic(p, u, B, G))
    u_da = _val(solve_optimal_dapts(p, u, B, G))
    u_mx = u_max(p, u)
    return {'U_single': u_s, 'U_s_NO': u_sno, 'U_s_O': u_so,
            'U_D': u_d, 'U_D_A': u_da, 'U_max': u_mx}


CHAIN = ['U_single', 'U_s_NO', 'U_s_O', 'U_D', 'U_D_A', 'U_max']


def run(configs, n_instances, base_seed=42, out_csv=None, tol=1e-9):
    rng = random.Random(base_seed)
    rows = []
    summary = []
    for (N, B, G) in configs:
        acc = {k: 0.0 for k in CHAIN}
        benefit_abs = 0.0
        benefit_pct = 0.0
        viol = 0
        for inst in range(n_instances):
            p = [rng.random() for _ in range(N)]
            u = [float(rng.choice((1, 2, 3))) for _ in range(N)]
            h = hierarchy_for_instance(p, u, B, G)
            # verify the chain
            for a, b in zip(CHAIN, CHAIN[1:]):
                if h[a] > h[b] + tol:
                    viol += 1
            for k in CHAIN:
                acc[k] += h[k]
            benefit_abs += h['U_D_A'] - h['U_D']
            if h['U_D'] > 1e-12:
                benefit_pct += (h['U_D_A'] - h['U_D']) / h['U_D'] * 100.0
            row = {'N': N, 'B': B, 'G': G, 'instance': inst, **h}
            rows.append(row)
        means = {k: acc[k] / n_instances for k in CHAIN}
        summary.append({
            'N': N, 'B': B, 'G': G, 'instances': n_instances,
            **{f'mean_{k}': means[k] for k in CHAIN},
            'benefit_abs': benefit_abs / n_instances,
            'benefit_pct': benefit_pct / n_instances,
            'chain_violations': viol,
        })

    # report
    print(f"\nHierarchy experiment — {n_instances} instances/config, "
          f"base_seed={base_seed}\n")
    hdr = "N  B  G | " + "  ".join(f"{k:>8}" for k in CHAIN) + \
          " | benefit(U_D_A-U_D)   %    chain_viol"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        means = "  ".join(f"{s['mean_'+k]:8.4f}" for k in CHAIN)
        print(f"{s['N']}  {s['B']}  {s['G']} | {means} | "
              f"{s['benefit_abs']:+8.4f}      {s['benefit_pct']:+6.2f}%   "
              f"{s['chain_violations']}")
    print()

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        spath = out_csv.replace('.csv', '_summary.csv')
        with open(spath, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"wrote {out_csv} and {spath}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instances', type=int, default=200)
    ap.add_argument('--quick', action='store_true',
                    help='1 instance per config, smoke/timing only')
    ap.add_argument('--with-n7', action='store_true',
                    help='also run the heavier N=G=7,B=3 curve point')
    ap.add_argument('--out', default='results/hierarchy/hierarchy.csv')
    args = ap.parse_args(argv)

    configs = [(3, 2, 3), (5, 3, 5)]
    if args.with_n7:
        configs.append((7, 3, 7))
    k = 1 if args.quick else args.instances
    run(configs, k, out_csv=None if args.quick else args.out)


if __name__ == "__main__":
    main()
