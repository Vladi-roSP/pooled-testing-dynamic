"""
Demo: two simple RL views of augmented DAPTS.

Prints:
  1. Value iteration on a tiny (n=2, B=1) instance, spelling out the
     Bellman backup on the initial state.
  2. Value iteration on n=3, B=2, checked against the brute-force DP.
  3. Tabular Q-learning on n=3, B=2, plotting the learning curve.

Run:
    python augmented/rl_examples_demo.py

Produces:
    augmented/figures/rl_q_learning_curve.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from augmented.core import all_pools, mask_str
from augmented.solver import solve_optimal_dapts
from augmented.rl_examples import (
    value_iteration_optimal_value,
    tabular_q_learning, q_learning_policy_value,
    _prior_weights, _cleared_utility, _transition,
)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _demo_tiny_vi():
    n, B, G = 2, 1, 2
    p = [0.3, 0.4]
    u = [1.0, 1.0]
    w = _prior_weights(p, n)
    pools = all_pools(n, G, include_empty=True)
    all_z = frozenset(range(1 << n))

    print("=== Ejemplo 1: n=2, B=1, G=2 ===")
    print(f"Prior p = {p}, utilidad u = {u}")
    print("\nAcciones (pool masks):")
    for a in pools:
        print(f"  a={mask_str(a, n)}")

    print("\nBellman sobre el estado inicial s0 = (k=0, remaining=all, cleared=0):")
    total_mass = sum(w[z] for z in all_z)
    for a in pools:
        ev = 0.0
        parts = []
        for r, mass_r, new_rem, new_cl in _transition(all_z, 0, a, w):
            prob = mass_r / total_mass
            util = _cleared_utility(new_cl, u, n)
            ev += prob * util
            parts.append(f"P(r={r})={prob:.3f} → cleared={mask_str(new_cl, n)} (u={util:.2f})")
        print(f"  Q*(s0, {mask_str(a, n)}) = {ev:.4f}    [{'; '.join(parts)}]")

    opt = value_iteration_optimal_value(p, u, B, G)
    print(f"\nV*(s0) = max Q* = {opt:.4f}")


def _demo_medium_vi():
    n, B, G = 3, 2, 3
    p = [0.2, 0.3, 0.4]
    u = [1.0, 2.0, 1.5]

    print("\n=== Ejemplo 2: n=3, B=2, G=3 ===")
    opt_dp, _ = solve_optimal_dapts(p, u, B, G)
    opt_vi = value_iteration_optimal_value(p, u, B, G)
    print(f"DP exacto (solve_optimal_dapts): {opt_dp:.6f}")
    print(f"Value iteration sobre MDP     : {opt_vi:.6f}")
    print(f"Diferencia                    : {abs(opt_dp - opt_vi):.2e}")


def _demo_q_learning():
    n, B, G = 3, 2, 3
    p = [0.2, 0.3, 0.4]
    u = [1.0, 2.0, 1.5]
    opt = value_iteration_optimal_value(p, u, B, G)

    print("\n=== Ejemplo 3: tabular Q-learning en n=3, B=2 ===")
    print(f"V* = {opt:.4f}  (meta que Q-learning debe aprender)")

    episode_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    seeds = list(range(6))

    curves = {s: [] for s in seeds}
    for seed in seeds:
        for ep in episode_counts:
            Q = tabular_q_learning(p, u, B, G, num_episodes=ep,
                                    epsilon=0.5, seed=seed)
            val = q_learning_policy_value(p, u, B, G, Q)
            curves[seed].append(val)

    for seed in seeds:
        print(f"  seed={seed}: "
              + "  ".join(f"ep={ep:>5}→{v:.3f}"
                          for ep, v in zip(episode_counts, curves[seed])))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, ys in curves.items():
        ax.plot(episode_counts, ys, marker='o', alpha=0.6,
                label=f"seed {seed}")
    ax.axhline(opt, color='k', linestyle='--', label=f"V* = {opt:.3f}")
    ax.set_xscale('log')
    ax.set_xlabel("episodios de entrenamiento")
    ax.set_ylabel("valor de la politica aprendida")
    ax.set_title("Q-learning converge a V* (n=3, B=2)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    outpath = os.path.join(FIGDIR, "rl_q_learning_curve.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"\nsaved {outpath}")


def main():
    _demo_tiny_vi()
    _demo_medium_vi()
    _demo_q_learning()


if __name__ == "__main__":
    main()
