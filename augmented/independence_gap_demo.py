"""
Demo: how close is the independence heuristic to the true joint posterior
of a pool's outcome?

Run:
    python augmented/independence_gap_demo.py
Produces:
    augmented/figures/independence_gap_tv_by_size.png
    augmented/figures/independence_gap_endpoints.png
    prints a text summary to stdout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from augmented.core import mask_from_indices, mask_str
from augmented.independence_gap import (
    run_experiment, aggregate, exact_pool_pmf, independence_pool_pmf,
)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _print_worked_example():
    # Exactly the case the user described.
    n = 4
    p = [0.5, 0.5, 0.5, 0.5]
    tprime = mask_from_indices([0, 1])
    history = ((tprime, 1),)   # t' returns "1 infected among {0,1}"
    t = mask_from_indices([0, 1, 2, 3])

    exact = exact_pool_pmf(p, history, t, n)
    heur = independence_pool_pmf(p, history, t, n)
    print("\n=== Worked example: t'={0,1} with r'=1, t={0,1,2,3}, "
          "prior 0.5 each ===")
    print(f"exact P(r_t = k | H): {[f'{v:.4f}' for v in exact]}")
    print(f"heuristic (indep.):   {[f'{v:.4f}' for v in heur]}")
    print(f"gap at r=0 (true 0, heur > 0): "
          f"{heur[0] - exact[0]:.4f}")
    print(f"TV distance:          {0.5 * sum(abs(a-b) for a,b in zip(exact, heur)):.4f}")


def _print_aggregate(rows):
    print("\n=== Aggregate gap statistics, by pool size ===")
    stats = aggregate(rows)
    hdr = (f"{'|t|':>4} {'count':>6} {'mean TV':>9} {'median TV':>10} "
           f"{'p95 TV':>8} {'max TV':>8} {'mean |r0 gap|':>14} "
           f"{'mean |rmax gap|':>16}")
    print(hdr)
    print("-" * len(hdr))
    for size, s in stats.items():
        print(f"{size:>4} {s['count']:>6} {s['tv_mean']:>9.4f} "
              f"{s['tv_median']:>10.4f} {s['tv_p95']:>8.4f} "
              f"{s['tv_max']:>8.4f} {s['abs_gap_r0_mean']:>14.4f} "
              f"{s['abs_gap_rmax_mean']:>16.4f}")


def _print_worst_cases(rows, k=5):
    print(f"\n=== Top {k} worst gaps (largest TV) ===")
    top = sorted(rows, key=lambda r: -r['tv'])[:k]
    for r in top:
        print(f"TV={r['tv']:.4f}  |t|={r['pool_size']}  "
              f"pool={mask_str(r['pool_mask'])}  "
              f"P(r=0) exact={r['exact_pmf'][0]:.3f} "
              f"heur={r['heuristic_pmf'][0]:.3f}  "
              f"prior_mean={r['prior_mean']:.2f}")


def _plot_tv_by_size(rows, path):
    by_size = {}
    for r in rows:
        by_size.setdefault(r['pool_size'], []).append(r['tv'])
    sizes = sorted(by_size)
    data = [by_size[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=[str(s) for s in sizes], showfliers=True)
    ax.set_xlabel("pool size |t|")
    ax.set_ylabel("TV(exact PMF, independence PMF)")
    ax.set_title("Gap between true joint posterior and product-of-marginals")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def _plot_endpoints(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter([r['exact_pmf'][0] for r in rows],
                    [r['heuristic_pmf'][0] for r in rows],
                    s=8, alpha=0.4)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel("exact P(r=0 | H)")
    axes[0].set_ylabel("heuristic ∏(1 - tilde p_i)")
    axes[0].set_title("All-healthy endpoint")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].scatter([r['exact_pmf'][r['pool_size']] for r in rows],
                    [r['heuristic_pmf'][r['pool_size']] for r in rows],
                    s=8, alpha=0.4, color='C3')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel("exact P(r=|t| | H)")
    axes[1].set_ylabel("heuristic ∏ tilde p_i")
    axes[1].set_title("All-infected endpoint")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Heuristic vs. exact posterior at the two endpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main():
    _print_worked_example()

    print("\nRunning experiment (n=8, B=3, G=3, 300 instances, greedy history)...")
    rows = run_experiment(n=8, B=3, G=3, num_instances=300, seed=0,
                          history_strategy='greedy')
    _print_aggregate(rows)
    _print_worst_cases(rows)

    _plot_tv_by_size(rows, os.path.join(FIGDIR, "independence_gap_tv_by_size.png"))
    _plot_endpoints(rows, os.path.join(FIGDIR, "independence_gap_endpoints.png"))


if __name__ == "__main__":
    main()
