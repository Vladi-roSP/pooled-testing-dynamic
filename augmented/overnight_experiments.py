"""
Overnight experiment runner for large-n instances.

Designed to avoid memory saturation:
  1. Uses greedy_myopic_expected_utility (value only, no tree storage)
  2. Writes results to CSV incrementally (no accumulation in RAM)
  3. Explicit gc.collect() between runs
  4. Optional memory limit via resource module

Usage:
    python augmented/overnight_experiments.py
    python augmented/overnight_experiments.py --n 30 50 80 --B 2 3 --G 5
    nohup python augmented/overnight_experiments.py > overnight.log 2>&1 &

Results saved to: results/overnight_YYYY-MM-DD_HHMMSS.csv
"""

import sys, os, gc, time, csv, argparse, random, resource
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.baselines import u_max, u_single
from augmented.greedy import greedy_myopic_expected_utility
from augmented.pool_solvers import gurobi_best_pool, mosek_best_pool


# -------------------------------------------------------------------
# Memory management
# -------------------------------------------------------------------

def get_memory_mb():
    """Current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def set_memory_limit_gb(limit_gb):
    """Set soft memory limit (macOS/Linux). Process gets killed if exceeded."""
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        print(f"  Memory limit set to {limit_gb:.1f} GB")
    except (ValueError, resource.error):
        # macOS doesn't always support RLIMIT_AS
        print(f"  Warning: could not set memory limit (OS limitation)")


# -------------------------------------------------------------------
# Core experiment
# -------------------------------------------------------------------

def run_single_instance(p, u, B, G, solver='gurobi', timeout_per_step=30):
    """Run greedy with solver pool selection. Returns EU value only (no tree).

    This is memory-efficient: greedy_myopic_expected_utility only keeps
    the recursion stack in memory, no tree dicts.
    """
    n = len(p)

    if solver == 'gurobi':
        pool_fn = gurobi_best_pool
    elif solver == 'mosek':
        pool_fn = mosek_best_pool
    else:
        pool_fn = None  # default enumeration

    results = {}

    # Upper/lower bounds (cheap)
    results['U_max'] = u_max(p, u)
    results['U_single'], _ = u_single(p, u, B)

    # Solver-based greedy
    t0 = time.time()
    results['U_greedy_solver'] = greedy_myopic_expected_utility(
        p, u, B, G, pool_selector=pool_fn
    )
    results['time_solver'] = time.time() - t0

    # Default greedy (enumeration) — only for small n where feasible
    if n <= 20:
        t0 = time.time()
        results['U_greedy_enum'] = greedy_myopic_expected_utility(p, u, B, G)
        results['time_enum'] = time.time() - t0
    else:
        results['U_greedy_enum'] = None
        results['time_enum'] = None

    return results


def generate_instance(n, B, G, p_range, u_range, seed):
    """Generate a random instance."""
    rng = random.Random(seed)
    p = [rng.uniform(*p_range) for _ in range(n)]
    u = [rng.uniform(*u_range) for _ in range(n)]
    return p, u


# -------------------------------------------------------------------
# Main experiment loop
# -------------------------------------------------------------------

def run_experiments(n_values, B_values, G_values, n_instances=10,
                    p_ranges=None, u_range=(1.0, 10.0),
                    solver='gurobi', base_seed=42,
                    memory_limit_gb=None, output_dir='results'):
    """Run experiments and write results incrementally to CSV."""

    if p_ranges is None:
        p_ranges = [
            ("low", (0.01, 0.10)),
            ("medium", (0.10, 0.30)),
            ("high", (0.30, 0.60)),
        ]

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"overnight_{timestamp}.csv")

    fieldnames = [
        'n', 'B', 'G', 'regime', 'instance', 'seed', 'solver',
        'U_max', 'U_single', 'U_greedy_solver', 'U_greedy_enum',
        'time_solver', 'time_enum', 'memory_mb',
    ]

    # Set memory limit if requested
    if memory_limit_gb:
        set_memory_limit_gb(memory_limit_gb)

    total_runs = (len(n_values) * len(B_values) * len(G_values)
                  * len(p_ranges) * n_instances)
    run_idx = 0

    print(f"Starting {total_runs} experiments → {csv_path}")
    print(f"Solver: {solver}")
    print(f"n ∈ {n_values}, B ∈ {B_values}, G ∈ {G_values}")
    print(f"Regimes: {[r[0] for r in p_ranges]}")
    print(f"{n_instances} instances per configuration")
    print()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for n in n_values:
            for B in B_values:
                for G in G_values:
                    if G > n:
                        continue
                    for regime_name, p_range in p_ranges:
                        for inst_idx in range(n_instances):
                            run_idx += 1
                            seed = base_seed + inst_idx

                            p, u = generate_instance(
                                n, B, G, p_range, u_range, seed
                            )

                            try:
                                results = run_single_instance(
                                    p, u, B, G, solver=solver
                                )
                            except MemoryError:
                                print(f"  MemoryError at n={n}, B={B}, "
                                      f"G={G}, {regime_name} #{inst_idx}")
                                results = {
                                    'U_max': None, 'U_single': None,
                                    'U_greedy_solver': None,
                                    'U_greedy_enum': None,
                                    'time_solver': None, 'time_enum': None,
                                }
                                gc.collect()

                            mem_mb = get_memory_mb()

                            row = {
                                'n': n, 'B': B, 'G': G,
                                'regime': regime_name,
                                'instance': inst_idx,
                                'seed': seed,
                                'solver': solver,
                                'U_max': results.get('U_max'),
                                'U_single': results.get('U_single'),
                                'U_greedy_solver': results.get('U_greedy_solver'),
                                'U_greedy_enum': results.get('U_greedy_enum'),
                                'time_solver': results.get('time_solver'),
                                'time_enum': results.get('time_enum'),
                                'memory_mb': f"{mem_mb:.1f}",
                            }
                            writer.writerow(row)
                            f.flush()  # write to disk immediately

                            # Progress
                            pct = run_idx / total_runs * 100
                            solver_time = results.get('time_solver')
                            time_str = (f"{solver_time:.1f}s"
                                        if solver_time else "ERR")
                            print(
                                f"  [{run_idx}/{total_runs} {pct:.0f}%] "
                                f"n={n} B={B} G={G} {regime_name:6s} "
                                f"#{inst_idx} → {time_str}  "
                                f"(mem: {mem_mb:.0f} MB)",
                                flush=True,
                            )

                            # Free memory between runs
                            del p, u, results
                            gc.collect()

    print(f"\nDone. Results in {csv_path}")
    return csv_path


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overnight large-n experiments for pooled testing"
    )
    parser.add_argument(
        '--n', nargs='+', type=int,
        default=[10, 20, 30, 50],
        help='Population sizes to test (default: 10 20 30 50)',
    )
    parser.add_argument(
        '--B', nargs='+', type=int,
        default=[2, 3],
        help='Test budgets (default: 2 3)',
    )
    parser.add_argument(
        '--G', nargs='+', type=int,
        default=[3, 5],
        help='Max pool sizes (default: 3 5)',
    )
    parser.add_argument(
        '--instances', type=int, default=10,
        help='Number of random instances per config (default: 10)',
    )
    parser.add_argument(
        '--solver', choices=['gurobi', 'mosek', 'enum'],
        default='gurobi',
        help='Pool selection solver (default: gurobi)',
    )
    parser.add_argument(
        '--memory-limit', type=float, default=None,
        help='Memory limit in GB (default: no limit)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed (default: 42)',
    )

    args = parser.parse_args()

    run_experiments(
        n_values=args.n,
        B_values=args.B,
        G_values=args.G,
        n_instances=args.instances,
        solver=args.solver,
        base_seed=args.seed,
        memory_limit_gb=args.memory_limit,
    )


if __name__ == "__main__":
    main()
