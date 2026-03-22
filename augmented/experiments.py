"""
Random instance exploration for augmented pooled testing.

Generates random problem instances and compares all strategies,
with a focus on how augmented testing performance varies with
infection rates (Francisco's hypothesis: augmented testing gains
increase at higher infection rates).

Usage:  python augmented/experiments.py
"""

import sys
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.classical_solver import solve_classical_dynamic
from augmented.greedy import (greedy_myopic_expected_utility,
                              greedy_myopic_counting_expected_utility)
from augmented.pool_solvers import mosek_best_pool, gurobi_best_pool


def random_instance(n, B, G, p_range=(0.01, 0.50), u_range=(1.0, 10.0),
                    seed=None):
    """Generate a random problem instance.

    Parameters
    ----------
    n : int
        Population size.
    B : int
        Test budget.
    G : int
        Max pool size.
    p_range : tuple
        (min, max) for infection probabilities.
    u_range : tuple
        (min, max) for individual utilities.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys: p, u, B, G, n, seed
    """
    rng = random.Random(seed)
    p = [rng.uniform(*p_range) for _ in range(n)]
    u = [rng.uniform(*u_range) for _ in range(n)]
    return {'p': p, 'u': u, 'B': B, 'G': G, 'n': n, 'seed': seed}


def evaluate_instance(instance, include_counting_greedy=True):
    """Run all strategies on a single instance.

    Returns dict of strategy names -> expected utility values.
    """
    p, u, B, G = instance['p'], instance['u'], instance['B'], instance['G']
    n = instance['n']

    results = {}
    results['U_max'] = u_max(p, u)
    results['U_single'], _ = u_single(p, u, B)

    # Optimal solvers (may be slow for large n)
    if n <= 14:
        results['U_D'], _ = solve_classical_dynamic(p, u, B, G)
        results['U_D_A'], _ = solve_optimal_dapts(p, u, B, G)

    # Greedy strategies
    results['U_greedy'] = greedy_myopic_expected_utility(p, u, B, G)

    if include_counting_greedy:
        results['U_greedy_counting'] = greedy_myopic_counting_expected_utility(
            p, u, B, G)

    # Solver-based greedy (works for any n)
    results['U_greedy_mosek'] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=mosek_best_pool)
    results['U_greedy_gurobi'] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=gurobi_best_pool)

    return results


def run_experiment(n_instances=50, n=5, B=2, G=3,
                   p_ranges=None, u_range=(1.0, 10.0),
                   base_seed=42, include_counting_greedy=True):
    """Run experiments across multiple random instances.

    Parameters
    ----------
    n_instances : int
        Number of random instances per infection-rate regime.
    n : int
        Population size.
    B : int
        Test budget.
    G : int
        Max pool size.
    p_ranges : list of (float, float) or None
        Infection probability ranges to test.
        Default: low (0.01-0.10), medium (0.10-0.30), high (0.30-0.60).
    u_range : tuple
        Utility range.
    base_seed : int
        Base random seed.
    include_counting_greedy : bool
        Whether to include the counting-based greedy.

    Returns
    -------
    dict mapping regime_name -> list of (instance, results) pairs
    """
    if p_ranges is None:
        p_ranges = [
            ("low (0.01-0.10)", (0.01, 0.10)),
            ("medium (0.10-0.30)", (0.10, 0.30)),
            ("high (0.30-0.60)", (0.30, 0.60)),
        ]

    all_results = {}
    for regime_name, p_range in p_ranges:
        regime_results = []
        for i in range(n_instances):
            seed = base_seed + i
            inst = random_instance(n, B, G, p_range=p_range,
                                   u_range=u_range, seed=seed)
            res = evaluate_instance(inst, include_counting_greedy)
            regime_results.append((inst, res))
        all_results[regime_name] = regime_results

    return all_results


def summarize_results(all_results):
    """Print summary statistics for experiment results.

    Shows mean, std, and relative performance of each strategy,
    with focus on augmented benefit at different infection rates.
    """
    print(f"\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for regime_name, regime_data in all_results.items():
        print(f"\n  Regime: {regime_name}")
        print(f"  {'─'*60}")

        # Collect all strategy names
        all_keys = set()
        for _, res in regime_data:
            all_keys.update(res.keys())

        # Compute statistics
        stats = {}
        for key in sorted(all_keys):
            values = [res[key] for _, res in regime_data if key in res]
            if values:
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / len(values)
                std = var ** 0.5
                stats[key] = {'mean': mean, 'std': std, 'count': len(values)}

        # Print strategy performance
        labels = {
            'U_max': 'U_max (upper bound)',
            'U_single': 'U_single (individual)',
            'U_D': 'U_D (classical dynamic)',
            'U_D_A': 'U_D_A (augmented dynamic)',
            'U_greedy': 'U_greedy (myopic)',
            'U_greedy_counting': 'U_greedy_counting (full-history)',
            'U_greedy_mosek': 'U_greedy_mosek (Mosek solver)',
            'U_greedy_gurobi': 'U_greedy_gurobi (Gurobi solver)',
        }

        order = ['U_single', 'U_D', 'U_D_A', 'U_greedy',
                 'U_greedy_counting', 'U_greedy_mosek', 'U_greedy_gurobi',
                 'U_max']
        for key in order:
            if key in stats:
                s = stats[key]
                label = labels.get(key, key)
                print(f"    {label:40s} "
                      f"mean={s['mean']:.4f}  std={s['std']:.4f}")

        # Augmented benefit over classical
        if 'U_D' in stats and 'U_D_A' in stats:
            benefits = []
            for _, res in regime_data:
                if 'U_D' in res and 'U_D_A' in res and res['U_D'] > 1e-10:
                    benefit = (res['U_D_A'] - res['U_D']) / res['U_D'] * 100
                    benefits.append(benefit)
            if benefits:
                mean_b = sum(benefits) / len(benefits)
                max_b = max(benefits)
                print(f"\n    Augmented benefit over classical:")
                print(f"      mean = +{mean_b:.2f}%,  max = +{max_b:.2f}%")

        # Greedy vs optimal gap
        if 'U_D_A' in stats and 'U_greedy' in stats:
            gaps = []
            for _, res in regime_data:
                if 'U_D_A' in res and 'U_greedy' in res and res['U_D_A'] > 1e-10:
                    gap = (res['U_D_A'] - res['U_greedy']) / res['U_D_A'] * 100
                    gaps.append(gap)
            if gaps:
                mean_g = sum(gaps) / len(gaps)
                max_g = max(gaps)
                print(f"    Greedy optimality gap:")
                print(f"      mean = {mean_g:.2f}%,  max = {max_g:.2f}%")


def find_best_instances(all_results, metric='augmented_benefit', top_k=5):
    """Find the instances where augmented testing helps the most.

    Parameters
    ----------
    all_results : dict
        From run_experiment().
    metric : str
        'augmented_benefit' or 'greedy_gap'.
    top_k : int
        Number of top instances to return.

    Returns
    -------
    list of (benefit, instance, results) sorted descending.
    """
    scored = []
    for regime_name, regime_data in all_results.items():
        for inst, res in regime_data:
            if metric == 'augmented_benefit':
                if 'U_D' in res and 'U_D_A' in res and res['U_D'] > 1e-10:
                    score = (res['U_D_A'] - res['U_D']) / res['U_D'] * 100
                    scored.append((score, inst, res, regime_name))
            elif metric == 'greedy_gap':
                if 'U_D_A' in res and 'U_greedy' in res and res['U_D_A'] > 1e-10:
                    score = (res['U_D_A'] - res['U_greedy']) / res['U_D_A'] * 100
                    scored.append((score, inst, res, regime_name))

    scored.sort(reverse=True)
    return scored[:top_k]


def main():
    print("Running random instance experiments...")
    print("n=5, B=2, G=3, 20 instances per regime\n")

    t0 = time.time()
    results = run_experiment(
        n_instances=20, n=5, B=2, G=3,
        include_counting_greedy=True,
        base_seed=42,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    summarize_results(results)

    # Find best instances for augmented benefit
    print(f"\n{'='*70}")
    print("  TOP 5 INSTANCES: AUGMENTED BENEFIT")
    print(f"{'='*70}")
    top = find_best_instances(results, metric='augmented_benefit', top_k=5)
    for i, (score, inst, res, regime) in enumerate(top):
        print(f"\n  #{i+1}: +{score:.2f}% augmented benefit ({regime})")
        print(f"    p = [{', '.join(f'{pi:.3f}' for pi in inst['p'])}]")
        print(f"    u = [{', '.join(f'{ui:.1f}' for ui in inst['u'])}]")
        print(f"    U_D = {res['U_D']:.4f}, U_D_A = {res['U_D_A']:.4f}")


if __name__ == "__main__":
    main()
