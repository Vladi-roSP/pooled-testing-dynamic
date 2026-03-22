"""
Comparison framework: compute all 6 strategies from Section 2.1 + greedy.

Usage:  python augmented/comparison.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.baselines import u_max, u_single
from augmented.static_solver import solve_static_non_overlapping, solve_static_overlapping
from augmented.classical_solver import solve_classical_dynamic
from augmented.solver import solve_optimal_dapts
from augmented.greedy import (greedy_myopic_expected_utility,
                              greedy_myopic_counting_expected_utility,
                              greedy_myopic_gibbs_expected_utility)
from augmented.pool_solvers import mosek_best_pool, gurobi_best_pool


def compare_all(p, u, B, G):
    """Compute all strategy values for population (p, u) with budget B and pool size G.

    Returns dict with keys: U_single, U_s_NO, U_s_O, U_D, U_D_A,
    U_greedy, U_greedy_counting, U_max.
    """
    n = len(p)
    results = {}

    results["U_single"], _ = u_single(p, u, B)
    results["U_s_NO"], _ = solve_static_non_overlapping(p, u, B, G)
    results["U_s_O"], _ = solve_static_overlapping(p, u, B, G)
    results["U_D"], _ = solve_classical_dynamic(p, u, B, G)
    results["U_D_A"], _ = solve_optimal_dapts(p, u, B, G)
    results["U_greedy"] = greedy_myopic_expected_utility(p, u, B, G)
    results["U_greedy_counting"] = greedy_myopic_counting_expected_utility(p, u, B, G)
    results["U_greedy_gibbs"] = greedy_myopic_gibbs_expected_utility(p, u, B, G, seed=42)
    results["U_greedy_mosek"] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=mosek_best_pool)
    results["U_greedy_gurobi"] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=gurobi_best_pool)
    results["U_max"] = u_max(p, u)

    return results


def print_comparison(p, u, B, G, label=""):
    """Print formatted comparison table."""
    n = len(p)
    results = compare_all(p, u, B, G)

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  n={n}, B={B}, G={G}")
    print(f"  p = {p}")
    print(f"  u = {u}")
    print()

    # Inequality chain
    chain = ["U_single", "U_s_NO", "U_s_O", "U_D", "U_D_A", "U_max"]
    labels = {
        "U_single": "U^single  (individual tests)",
        "U_s_NO":   "U^s_NO    (static non-overlapping)",
        "U_s_O":    "U^s_O     (static overlapping)",
        "U_D":      "U^D       (dynamic classical)",
        "U_D_A":    "U^D_A     (dynamic augmented)",
        "U_max":    "U^max     (upper bound)",
    }

    for key in chain:
        print(f"  {labels[key]:42s} = {results[key]:.6f}")

    print()
    print(f"  {'U^greedy  (myopic augmented greedy)':42s} = {results['U_greedy']:.6f}")
    print(f"  {'U^greedy_counting  (full-history Bayes)':42s} = {results['U_greedy_counting']:.6f}")
    print(f"  {'U^greedy_gibbs  (Gibbs sampling MCMC)':42s} = {results['U_greedy_gibbs']:.6f}")
    print(f"  {'U^greedy_mosek  (Mosek pool selector)':42s} = {results['U_greedy_mosek']:.6f}")
    print(f"  {'U^greedy_gurobi (Gurobi pool selector)':42s} = {results['U_greedy_gurobi']:.6f}")

    # Verify inequality chain
    print()
    ok = True
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        if results[a] > results[b] + 1e-9:
            print(f"  WARNING: {a} = {results[a]:.6f} > {b} = {results[b]:.6f}")
            ok = False
    if ok:
        vals = " <= ".join(f"{results[k]:.4f}" for k in chain)
        print(f"  Inequality chain verified: {vals}")

    # Augmented benefit
    if results["U_D"] > 1e-10:
        benefit = (results["U_D_A"] - results["U_D"]) / results["U_D"] * 100
        print(f"  Augmented benefit over classical dynamic: +{benefit:.2f}%")

    return results


def main():
    # --- Instance 1: the running example ---
    print_comparison(
        p=[0.05, 0.10, 0.15, 0.20, 0.08],
        u=[4.0,  6.0,  3.0,  5.0,  7.0],
        B=2, G=3,
        label="Instance 1: n=5, B=2, G=3 (low infection rates)"
    )

    # --- Instance 2: higher infection rates ---
    print_comparison(
        p=[0.30, 0.40, 0.35, 0.25],
        u=[5.0,  3.0,  4.0,  6.0],
        B=2, G=2,
        label="Instance 2: n=4, B=2, G=2 (high infection rates)"
    )

    # --- Instance 3: very small ---
    print_comparison(
        p=[0.1, 0.2, 0.3],
        u=[5.0, 3.0, 4.0],
        B=2, G=2,
        label="Instance 3: n=3, B=2, G=2"
    )

    # --- Instance 4: uniform population ---
    print_comparison(
        p=[0.15, 0.15, 0.15, 0.15, 0.15],
        u=[1.0,  1.0,  1.0,  1.0,  1.0],
        B=2, G=3,
        label="Instance 4: n=5, B=2, G=3 (uniform population)"
    )


if __name__ == "__main__":
    main()
