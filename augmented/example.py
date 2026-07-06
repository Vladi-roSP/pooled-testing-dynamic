"""
Example: compare optimal DAPTS vs greedy strategies.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask, mask_from_indices, mask_str, popcount
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.simulator import apply_dapts
from augmented.greedy import (
    greedy_myopic_simulate, greedy_myopic_expected_utility,
    greedy_lookahead_simulate,
)
from augmented.bayesian import bayesian_update_single_test


def main():
    p = [0.05, 0.10, 0.15, 0.20, 0.08]
    u = [4.0,  6.0,  3.0,  5.0,  7.0]
    n = len(p)
    B = 2
    G = 3

    print("=" * 60)
    print("Dynamic Augmented Adaptive Group Counting — Comparison")
    print("=" * 60)
    print(f"\nPopulation (n={n}):")
    for i in range(n):
        print(f"  Individual {i}:  u={u[i]:.1f},  p={p[i]:.2f}")
    print(f"\nBudget B={B},  Max pool size G={G}")

    # --- All strategies ---
    um = u_max(p, u)
    us, _ = u_single(p, u, B)
    u_greedy = greedy_myopic_expected_utility(p, u, B, G)
    u_opt, F_opt = solve_optimal_dapts(p, u, B, G)

    print(f"\n--- Expected Utility Comparison ---")
    print(f"  U_single   = {us:.4f}  (test individuals one at a time)")
    print(f"  U_greedy   = {u_greedy:.4f}  (myopic greedy with Bayesian updates)")
    print(f"  U_optimal  = {u_opt:.4f}  (brute-force DP)")
    print(f"  U_max      = {um:.4f}  (upper bound)")
    print(f"\n  Greedy captures {(u_greedy - us) / (u_opt - us) * 100:.1f}%"
          f" of the gap between single and optimal")

    # --- Simulate on specific profiles ---
    print(f"\n--- Simulation: Z = {{3}} (only individual 3 active) ---")
    z = mask_from_indices([3])

    # Optimal
    hist_opt, cl_opt, u_opt_z = apply_dapts(F_opt, z, n, u)
    print(f"\n  Optimal:")
    current_p = list(p)
    for step, (pool, r) in enumerate(hist_opt, 1):
        current_p = bayesian_update_single_test(current_p, pool, r, n)
        print(f"    Step {step}: test {mask_str(pool, n)}, r={r}"
              f"{'  -> CLEARED' if r == 0 else ''}")
    print(f"    Cleared: {indices_from_mask(cl_opt, n)},  Utility: {u_opt_z:.1f}")

    # Greedy myopic
    hist_gr, cl_gr, u_gr_z = greedy_myopic_simulate(p, u, B, G, z)
    print(f"\n  Greedy myopic:")
    current_p = list(p)
    for step, (pool, r) in enumerate(hist_gr, 1):
        current_p = bayesian_update_single_test(current_p, pool, r, n)
        print(f"    Step {step}: test {mask_str(pool, n)}, r={r}"
              f"{'  -> CLEARED' if r == 0 else ''}")
    print(f"    Cleared: {indices_from_mask(cl_gr, n)},  Utility: {u_gr_z:.1f}")

    # Greedy lookahead
    hist_la, cl_la, u_la_z = greedy_lookahead_simulate(p, u, B, G, z)
    print(f"\n  Greedy lookahead:")
    current_p = list(p)
    for step, (pool, r) in enumerate(hist_la, 1):
        current_p = bayesian_update_single_test(current_p, pool, r, n)
        print(f"    Step {step}: test {mask_str(pool, n)}, r={r}"
              f"{'  -> CLEARED' if r == 0 else ''}")
    print(f"    Cleared: {indices_from_mask(cl_la, n)},  Utility: {u_la_z:.1f}")

    print()


if __name__ == "__main__":
    main()
