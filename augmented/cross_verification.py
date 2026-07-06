"""
Cross-verification module: generate synthetic test instances and export
them in a format compatible with Nick's group-count code at
github.com/nrlopez03/group-count.

Workflow:
  1. generate_synthetic_instances()  -- create random instances
  2. export_instances_json()         -- save raw instances to JSON
  3. evaluate_and_export()           -- solve with our algorithms, save results
  4. comparison_protocol()           -- print instructions for running in Nick's code

Usage:  python augmented/cross_verification.py
"""

import json
import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.solver import solve_optimal_dapts
from augmented.greedy import (greedy_myopic_expected_utility,
                              greedy_myopic_counting_expected_utility,
                              greedy_myopic_gibbs_expected_utility)
from augmented.classical_solver import solve_classical_dynamic
from augmented.baselines import u_max, u_single


# -------------------------------------------------------------------
# 1. Synthetic instance generation
# -------------------------------------------------------------------

def generate_synthetic_instances(num_instances, n, B, G,
                                  p_range=(0.0, 1.0),
                                  u_values=None,
                                  seed=42):
    """Generate random test instances for cross-verification.

    Parameters
    ----------
    num_instances : int
        Number of instances to generate.
    n : int
        Population size (number of agents).
    B : int
        Test budget.
    G : int
        Maximum pool size per test.
    p_range : tuple of float
        (lo, hi) range for each agent's probability of being clearancey.
        Each p_i is drawn from U[lo, hi].
    u_values : list of int/float or None
        Discrete set of utility values. Each agent's u_i is drawn
        uniformly from this list.  Defaults to [1, 2, 3].
    seed : int
        Base random seed for reproducibility.  Instance k uses
        seed = base_seed + k so every instance is independently
        reproducible.

    Returns
    -------
    list of dict
        Each dict has keys:
          id            -- integer instance identifier (0-indexed)
          n, B, G       -- problem parameters
          agents        -- list of {id, utility, prob_clearancey}
          seed          -- the per-instance seed used
    """
    if u_values is None:
        u_values = [1, 2, 3]

    instances = []
    for k in range(num_instances):
        inst_seed = seed + k
        rng = random.Random(inst_seed)

        agents = []
        for i in range(n):
            p_i = rng.uniform(p_range[0], p_range[1])
            u_i = rng.choice(u_values)
            agents.append({
                "id": i,
                "utility": u_i,
                "prob_clearancey": round(p_i, 10),
            })

        instances.append({
            "id": k,
            "n": n,
            "B": B,
            "G": G,
            "agents": agents,
            "seed": inst_seed,
        })

    return instances


# -------------------------------------------------------------------
# 2. Export raw instances to JSON
# -------------------------------------------------------------------

def export_instances_json(instances, filepath):
    """Write instances to a JSON file.

    Parameters
    ----------
    instances : list of dict
        As returned by generate_synthetic_instances().
    filepath : str
        Destination path (will be created / overwritten).
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(instances, f, indent=2)

    print(f"Exported {len(instances)} instances to {filepath}")


# -------------------------------------------------------------------
# 3. Evaluate with our solvers and export results
# -------------------------------------------------------------------

def _extract_p_u(instance):
    """Extract p and u vectors from an instance dict.

    NOTE: our solvers use p_i = P(active), while the instance stores
    prob_clearancey.  Convert accordingly.
    """
    p = [1.0 - ag["prob_clearancey"] for ag in instance["agents"]]
    u = [float(ag["utility"]) for ag in instance["agents"]]
    return p, u


def evaluate_and_export(instances, filepath):
    """Solve every instance with our algorithms and export results.

    For each instance the following are computed:
      - U_D_A             (optimal augmented DAPTS, exact DP -- only if n <= 14)
      - U_D               (optimal classical dynamic, exact DP -- only if n <= 14)
      - U_greedy          (myopic greedy with sequential Bayesian updates)
      - U_greedy_counting (myopic greedy with full-history counting updates)
      - U_greedy_gibbs    (myopic greedy with Gibbs drawing updates)
      - U_max             (upper bound)
      - U_single          (individual counting baseline)

    Parameters
    ----------
    instances : list of dict
        As returned by generate_synthetic_instances().
    filepath : str
        Destination JSON path.
    """
    results = []
    total = len(instances)

    for idx, inst in enumerate(instances):
        p, u = _extract_p_u(inst)
        n = inst["n"]
        B = inst["B"]
        G = inst["G"]

        t0 = time.time()

        record = {
            "instance": inst,
            "results": {},
        }

        # Baselines
        record["results"]["U_max"] = u_max(p, u)
        val_single, _ = u_single(p, u, B)
        record["results"]["U_single"] = val_single

        # Exact solvers (feasible only for small n)
        if n <= 14:
            val_da, _ = solve_optimal_dapts(p, u, B, G)
            record["results"]["U_D_A"] = val_da

            val_d, _ = solve_classical_dynamic(p, u, B, G)
            record["results"]["U_D"] = val_d
        else:
            record["results"]["U_D_A"] = None
            record["results"]["U_D"] = None

        # Greedy heuristics
        record["results"]["U_greedy"] = greedy_myopic_expected_utility(p, u, B, G)
        record["results"]["U_greedy_counting"] = greedy_myopic_counting_expected_utility(p, u, B, G)
        record["results"]["U_greedy_gibbs"] = greedy_myopic_gibbs_expected_utility(
            p, u, B, G, seed=inst["seed"]
        )

        elapsed = time.time() - t0
        record["solve_time_s"] = round(elapsed, 4)

        results.append(record)

        print(f"  [{idx + 1}/{total}] instance {inst['id']}: "
              f"U_D_A={record['results'].get('U_D_A', 'N/A'):.6f}  "
              f"U_greedy={record['results']['U_greedy']:.6f}  "
              f"({elapsed:.2f}s)")

    # Write
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExported {len(results)} evaluated instances to {filepath}")
    return results


# -------------------------------------------------------------------
# 4. Comparison protocol
# -------------------------------------------------------------------

def comparison_protocol():
    """Return a string describing how to run the same instances in Nick's code.

    This serves as documentation so that a collaborator can reproduce the
    cross-verification end-to-end.
    """
    protocol = """\
=======================================================================
  Cross-Verification Protocol
  Our code  <-->  github.com/nrlopez03/group-count
=======================================================================

Step 1 -- Generate & export instances (this code)
------
    python augmented/cross_verification.py
    This produces two files in augmented/data/:
      * cross_verify_instances.json   (raw instances)
      * cross_verify_results.json     (instances + our solver outputs)

Step 2 -- Load instances in Nick's code
------
    import json
    with open("cross_verify_instances.json") as f:
        instances = json.load(f)

    Each instance is a dict with keys:
        id, n, B, G, agents (list of {id, utility, prob_clearancey}), seed

    To convert to Nick's format:
        for inst in instances:
            n = inst["n"]
            B = inst["B"]
            G = inst["G"]
            # NOTE: our prob_clearancey = 1 - p_active
            p_active = [1.0 - ag["prob_clearancey"] for ag in inst["agents"]]
            utilities  = [ag["utility"] for ag in inst["agents"]]
            # ... run Nick's solver with (p_active, utilities, B, G)

Step 3 -- Compare outputs
------
    Load our results from cross_verify_results.json and compare
    the computed expected utilities (U_D_A, U_greedy, etc.) against
    Nick's outputs for the same instances.

    Suggested tolerance: |ours - theirs| < 1e-6 for exact solvers,
    1e-3 for MCMC-based (Gibbs) values.

=======================================================================
"""
    return protocol


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Generate 50 instances with n=5, B=3, G=3
    print("Generating 50 synthetic instances (n=5, B=3, G=3) ...")
    instances = generate_synthetic_instances(
        num_instances=50,
        n=5,
        B=3,
        G=3,
        p_range=(0.0, 1.0),
        u_values=[1, 2, 3],
        seed=42,
    )

    # Export raw instances
    instances_path = os.path.join(data_dir, "cross_verify_instances.json")
    export_instances_json(instances, instances_path)

    # Evaluate with our solvers and export
    results_path = os.path.join(data_dir, "cross_verify_results.json")
    print("\nEvaluating instances with all solvers ...")
    evaluate_and_export(instances, results_path)

    # Print comparison protocol
    print(comparison_protocol())
