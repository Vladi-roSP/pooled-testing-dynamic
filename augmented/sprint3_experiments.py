"""
Sprint 3 large-scale experiments for augmented adaptive group counting.

Run with:
    python augmented/sprint3_experiments.py --quick
"""

import argparse
import csv
import gc
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.baselines import u_max, u_single
from augmented.greedy import greedy_myopic_expected_utility
from augmented.state_reward_greedy import greedy_myopic_beta_expected_utility
from augmented.pool_solvers import mosek_best_pool


CONFIGS = [
    ("A", 20, 5, 10, "two big pools"),
    ("B", 30, 5, 10, "medium scale"),
    # ("C", 50, 10, 10, "large scale"),  # DISABLED: ~6h per instance (B=10 tree too deep)
    ("C2", 50, 3, 10, "large scale (reduced B)"),  # feasible: B=3 -> 2^3=8 branches
    ("D", 20, 2, 10, "Francisco two big tests"),
]

REGIMES = [
    ("low", (0.01, 0.10)),
    ("medium", (0.10, 0.30)),
    ("high", (0.30, 0.60)),
]

TIMEOUT_SECONDS = 120.0


def _timestamped_csv_path(output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(output_dir, f"{prefix}_{timestamp}.csv")


def _timed_call(fn, timeout_seconds=TIMEOUT_SECONDS):
    t0 = time.time()
    try:
        value = fn()
        elapsed = time.time() - t0
        return value, elapsed, None, elapsed > timeout_seconds
    except Exception as exc:
        elapsed = time.time() - t0
        return None, elapsed, str(exc), False


def _warn_if_slow(metric_name, elapsed, context):
    if elapsed > TIMEOUT_SECONDS:
        print(
            f"  Warning: {metric_name} exceeded {TIMEOUT_SECONDS:.0f}s "
            f"({elapsed:.1f}s) for {context}",
            flush=True,
        )


def _generate_random_instance(n, p_range, u_range, seed):
    rng = random.Random(seed)
    p = [rng.uniform(*p_range) for _ in range(n)]
    u = [rng.uniform(*u_range) for _ in range(n)]
    return p, u


def _generate_vip_instance(n_vip, n_reg, p_vip_base, p_reg_base,
                           u_vip_range, u_reg_range, seed):
    rng = random.Random(seed)
    p_vip = [
        min(0.95, max(0.05, p_vip_base + rng.uniform(-0.05, 0.05)))
        for _ in range(n_vip)
    ]
    p_reg = [
        min(0.95, max(0.05, p_reg_base + rng.uniform(-0.05, 0.05)))
        for _ in range(n_reg)
    ]
    u_vip = [rng.uniform(*u_vip_range) for _ in range(n_vip)]
    u_reg = [rng.uniform(*u_reg_range) for _ in range(n_reg)]
    return p_vip + p_reg, u_vip + u_reg


def _generate_utility_modulation_instance(n, utility_distribution, seed):
    rng = random.Random(seed)
    p = [rng.uniform(0.1, 0.3) for _ in range(n)]
    if utility_distribution == "uniform":
        u = [1.0] * n
    elif utility_distribution == "skewed":
        u = [float(rng.choice([1, 5, 10])) for _ in range(n)]
    elif utility_distribution == "extreme":
        u = [float(rng.choice([1, 100])) for _ in range(n)]
    else:
        raise ValueError(f"Unknown utility distribution: {utility_distribution}")
    return p, u


def _measure_baselines(p, u, B):
    u_single_val, _ = u_single(p, u, B)
    return {
        "U_max": u_max(p, u),
        "U_single": u_single_val,
    }


def _measure_greedy_mosek(p, u, B, G):
    return _timed_call(
        lambda: greedy_myopic_expected_utility(
            p, u, B, G, pool_selector=mosek_best_pool)
    )


def _measure_greedy_enum(p, u, B, G):
    if len(p) > 20:
        return None, None, None, False
    return _timed_call(lambda: greedy_myopic_expected_utility(p, u, B, G))


def _measure_beta_greedy(p, u, B, G, beta=1.0, info_metric='entropy'):
    return _timed_call(
        lambda: greedy_myopic_beta_expected_utility(
            p, u, B, G, beta=beta, info_metric=info_metric)
    )


def _write_row(writer, handle, row):
    writer.writerow(row)
    handle.flush()


def _progress(run_idx, total_runs, context, mosek_val, mosek_time):
    u_str = "ERR" if mosek_val is None else f"{mosek_val:.2f}"
    t_str = "ERR" if mosek_time is None else f"{mosek_time:.1f}s"
    print(
        f"[run {run_idx}/{total_runs}] {context} U_mosek={u_str} ({t_str})",
        flush=True,
    )


def run_main_experiments(n_instances=50, output_dir='results',
                         configs=None, regimes=None, seed=42):
    if configs is None:
        configs = CONFIGS
    if regimes is None:
        regimes = REGIMES

    csv_path = _timestamped_csv_path(output_dir, "sprint3")
    fieldnames = [
        "config", "notes", "n", "B", "G", "regime", "instance", "seed",
        "U_max", "U_single", "U_greedy_mosek", "time_greedy_mosek",
        "U_greedy_enum", "time_greedy_enum", "U_beta_greedy",
        "time_beta_greedy", "error",
    ]

    total_runs = len(configs) * len(regimes) * n_instances
    run_idx = 0
    print(f"Writing main Sprint 3 results to {csv_path}")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        for label, n, B, G, notes in configs:
            for regime_name, p_range in regimes:
                for inst_idx in range(n_instances):
                    run_idx += 1
                    inst_seed = seed + inst_idx
                    context = (
                        f"config={label} n={n} B={B} G={G} "
                        f"regime={regime_name} inst={inst_idx}"
                    )
                    row = {
                        "config": label,
                        "notes": notes,
                        "n": n,
                        "B": B,
                        "G": G,
                        "regime": regime_name,
                        "instance": inst_idx,
                        "seed": inst_seed,
                        "U_max": None,
                        "U_single": None,
                        "U_greedy_mosek": None,
                        "time_greedy_mosek": None,
                        "U_greedy_enum": None,
                        "time_greedy_enum": None,
                        "U_beta_greedy": None,
                        "time_beta_greedy": None,
                        "error": None,
                    }
                    try:
                        p, u = _generate_random_instance(
                            n, p_range, (1.0, 10.0), inst_seed)
                        row.update(_measure_baselines(p, u, B))

                        val, elapsed, error, timed_out = _measure_greedy_mosek(p, u, B, G)
                        row["U_greedy_mosek"] = val
                        row["time_greedy_mosek"] = elapsed
                        if timed_out:
                            _warn_if_slow("U_greedy_mosek", elapsed, context)
                        if error:
                            print(f"  Error in U_greedy_mosek for {context}: {error}",
                                  flush=True)

                        val, elapsed, error, timed_out = _measure_greedy_enum(p, u, B, G)
                        row["U_greedy_enum"] = val
                        row["time_greedy_enum"] = elapsed
                        if timed_out:
                            _warn_if_slow("U_greedy_enum", elapsed, context)
                        if error:
                            print(f"  Error in U_greedy_enum for {context}: {error}",
                                  flush=True)

                        val, elapsed, error, timed_out = _measure_beta_greedy(
                            p, u, B, G, beta=1.0, info_metric='entropy')
                        row["U_beta_greedy"] = val
                        row["time_beta_greedy"] = elapsed
                        if timed_out:
                            _warn_if_slow("U_beta_greedy", elapsed, context)
                        if error:
                            print(f"  Error in U_beta_greedy for {context}: {error}",
                                  flush=True)
                    except Exception as exc:
                        row["error"] = str(exc)
                        print(f"  Error in {context}: {exc}", flush=True)

                    _write_row(writer, handle, row)
                    _progress(run_idx, total_runs, context,
                              row["U_greedy_mosek"], row["time_greedy_mosek"])
                    gc.collect()

    print(f"Done. Main results in {csv_path}")
    return csv_path


def run_vip_experiments(output_dir='results', n_instances_v1=20,
                        n_instances_v2=10, configs=None):
    csv_path = _timestamped_csv_path(output_dir, "sprint3_vip")
    fieldnames = [
        "config", "n", "B", "G", "instance", "seed",
        "U_max", "U_single", "U_greedy_mosek", "time_greedy_mosek",
        "U_beta_greedy", "time_beta_greedy", "error",
    ]
    if configs is None:
        configs = [
            ("V1", 8, 12, 6, 10, n_instances_v1),
            ("V2", 10, 20, 6, 10, n_instances_v2),
        ]
    total_runs = sum(instances for _, _, _, _, _, instances in configs)
    run_idx = 0
    print(f"Writing VIP Sprint 3 results to {csv_path}")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        for label, n_vip, n_reg, B, G, n_instances in configs:
            n = n_vip + n_reg
            for inst_idx in range(n_instances):
                run_idx += 1
                inst_seed = inst_idx
                context = f"config={label} n={n} B={B} G={G} inst={inst_idx}"
                row = {
                    "config": label,
                    "n": n,
                    "B": B,
                    "G": G,
                    "instance": inst_idx,
                    "seed": inst_seed,
                    "U_max": None,
                    "U_single": None,
                    "U_greedy_mosek": None,
                    "time_greedy_mosek": None,
                    "U_beta_greedy": None,
                    "time_beta_greedy": None,
                    "error": None,
                }
                try:
                    p, u = _generate_vip_instance(
                        n_vip, n_reg, 0.35, 0.1, (10.0, 10.0), (2.0, 2.0),
                        inst_seed,
                    )
                    row.update(_measure_baselines(p, u, B))

                    val, elapsed, error, timed_out = _measure_greedy_mosek(p, u, B, G)
                    row["U_greedy_mosek"] = val
                    row["time_greedy_mosek"] = elapsed
                    if timed_out:
                        _warn_if_slow("U_greedy_mosek", elapsed, context)
                    if error:
                        print(f"  Error in U_greedy_mosek for {context}: {error}",
                              flush=True)

                    val, elapsed, error, timed_out = _measure_beta_greedy(
                        p, u, B, G, beta=1.0, info_metric='entropy')
                    row["U_beta_greedy"] = val
                    row["time_beta_greedy"] = elapsed
                    if timed_out:
                        _warn_if_slow("U_beta_greedy", elapsed, context)
                    if error:
                        print(f"  Error in U_beta_greedy for {context}: {error}",
                              flush=True)
                except Exception as exc:
                    row["error"] = str(exc)
                    print(f"  Error in {context}: {exc}", flush=True)

                _write_row(writer, handle, row)
                _progress(run_idx, total_runs, context,
                          row["U_greedy_mosek"], row["time_greedy_mosek"])
                gc.collect()

    print(f"Done. VIP results in {csv_path}")
    return csv_path


def run_utility_modulation(output_dir='results', n_instances=20, distributions=None):
    csv_path = _timestamped_csv_path(output_dir, "sprint3_utility")
    fieldnames = [
        "utility_distribution", "n", "B", "G", "instance", "seed",
        "U_max", "U_single", "U_greedy_mosek", "time_greedy_mosek",
        "U_beta_greedy", "time_beta_greedy", "error",
    ]
    if distributions is None:
        distributions = ["uniform", "skewed", "extreme"]
    n, B, G = 20, 5, 10
    total_runs = len(distributions) * n_instances
    run_idx = 0
    print(f"Writing utility modulation results to {csv_path}")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        for utility_distribution in distributions:
            for inst_idx in range(n_instances):
                run_idx += 1
                inst_seed = inst_idx
                context = (
                    f"dist={utility_distribution} n={n} B={B} G={G} inst={inst_idx}"
                )
                row = {
                    "utility_distribution": utility_distribution,
                    "n": n,
                    "B": B,
                    "G": G,
                    "instance": inst_idx,
                    "seed": inst_seed,
                    "U_max": None,
                    "U_single": None,
                    "U_greedy_mosek": None,
                    "time_greedy_mosek": None,
                    "U_beta_greedy": None,
                    "time_beta_greedy": None,
                    "error": None,
                }
                try:
                    p, u = _generate_utility_modulation_instance(
                        n, utility_distribution, inst_seed)
                    row.update(_measure_baselines(p, u, B))

                    val, elapsed, error, timed_out = _measure_greedy_mosek(p, u, B, G)
                    row["U_greedy_mosek"] = val
                    row["time_greedy_mosek"] = elapsed
                    if timed_out:
                        _warn_if_slow("U_greedy_mosek", elapsed, context)
                    if error:
                        print(f"  Error in U_greedy_mosek for {context}: {error}",
                              flush=True)

                    val, elapsed, error, timed_out = _measure_beta_greedy(
                        p, u, B, G, beta=1.0, info_metric='entropy')
                    row["U_beta_greedy"] = val
                    row["time_beta_greedy"] = elapsed
                    if timed_out:
                        _warn_if_slow("U_beta_greedy", elapsed, context)
                    if error:
                        print(f"  Error in U_beta_greedy for {context}: {error}",
                              flush=True)
                except Exception as exc:
                    row["error"] = str(exc)
                    print(f"  Error in {context}: {exc}", flush=True)

                _write_row(writer, handle, row)
                _progress(run_idx, total_runs, context,
                          row["U_greedy_mosek"], row["time_greedy_mosek"])
                gc.collect()

    print(f"Done. Utility modulation results in {csv_path}")
    return csv_path


def run_large_G(output_dir='results', n_instances=20, g_values=None):
    csv_path = _timestamped_csv_path(output_dir, "sprint3_largeG")
    fieldnames = [
        "G", "n", "B", "instance", "seed",
        "U_max", "U_single", "U_greedy_mosek", "time_greedy_mosek",
        "gap", "error",
    ]
    n, B = 20, 2
    if g_values is None:
        g_values = [5, 10, 15, 20]
    total_runs = len(g_values) * n_instances
    run_idx = 0
    print(f"Writing large-G results to {csv_path}")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        for G in g_values:
            for inst_idx in range(n_instances):
                run_idx += 1
                inst_seed = inst_idx
                context = f"G={G} n={n} B={B} inst={inst_idx}"
                row = {
                    "G": G,
                    "n": n,
                    "B": B,
                    "instance": inst_idx,
                    "seed": inst_seed,
                    "U_max": None,
                    "U_single": None,
                    "U_greedy_mosek": None,
                    "time_greedy_mosek": None,
                    "gap": None,
                    "error": None,
                }
                try:
                    p, u = _generate_random_instance(n, (0.1, 0.3), (1.0, 10.0), inst_seed)
                    row.update(_measure_baselines(p, u, B))

                    val, elapsed, error, timed_out = _measure_greedy_mosek(p, u, B, G)
                    row["U_greedy_mosek"] = val
                    row["time_greedy_mosek"] = elapsed
                    if timed_out:
                        _warn_if_slow("U_greedy_mosek", elapsed, context)
                    if error:
                        print(f"  Error in U_greedy_mosek for {context}: {error}",
                              flush=True)
                    if row["U_max"] and row["U_greedy_mosek"] is not None:
                        row["gap"] = (
                            (row["U_max"] - row["U_greedy_mosek"]) / row["U_max"]
                        )
                except Exception as exc:
                    row["error"] = str(exc)
                    print(f"  Error in {context}: {exc}", flush=True)

                _write_row(writer, handle, row)
                _progress(run_idx, total_runs, context,
                          row["U_greedy_mosek"], row["time_greedy_mosek"])
                gc.collect()

    print(f"Done. Large-G results in {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs',
        choices=['main', 'vip', 'utility', 'largeG', 'all'],
        default='all',
    )
    parser.add_argument('--n-instances', type=int, default=50)
    parser.add_argument(
        '--quick', action='store_true',
        help='Run 3 instances for counting',
    )
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    n_inst = 3 if args.quick else args.n_instances

    main_configs = CONFIGS
    main_regimes = REGIMES
    vip_configs = None
    utility_distributions = None
    large_g_values = None
    vip_v1 = 20
    vip_v2 = 10

    if args.quick:
        n_inst = 1
        main_configs = [CONFIGS[3]]
        main_regimes = [REGIMES[1]]
        vip_configs = [("V1", 8, 12, 6, 10, 1)]
        utility_distributions = ["uniform", "skewed"]
        large_g_values = [5, 10]
        vip_v1 = 1
        vip_v2 = 1

    if args.configs in ('main', 'all'):
        run_main_experiments(
            n_instances=n_inst,
            output_dir=args.output_dir,
            configs=main_configs,
            regimes=main_regimes,
        )
    if args.configs in ('vip', 'all'):
        run_vip_experiments(
            output_dir=args.output_dir,
            n_instances_v1=vip_v1 if args.quick else n_inst,
            n_instances_v2=vip_v2 if args.quick else n_inst,
            configs=vip_configs,
        )
    if args.configs in ('utility', 'all'):
        run_utility_modulation(
            output_dir=args.output_dir,
            n_instances=n_inst,
            distributions=utility_distributions,
        )
    if args.configs in ('largeG', 'all'):
        run_large_G(
            output_dir=args.output_dir,
            n_instances=n_inst,
            g_values=large_g_values,
        )


if __name__ == '__main__':
    main()
