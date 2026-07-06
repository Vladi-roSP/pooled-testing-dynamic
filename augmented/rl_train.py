"""
Reproducible PPO training and evaluation for the augmented DAPTS RL
environments (augmented/rl_env.py).

This is the augmented-package counterpart of
``classical/rl_training/PPO_bucket_gymnasium_B*.py``: same PPO / Gymnasium
recipe, but powered by augmented Bayesian updates and myopic greedy (no MOSEK),
fully seeded, and with an exact-vs-DP validation path.

Instance sources
----------------
  random : (p, u) drawn from a fixed-seed RNG  -> 100% reproducible.
  csv    : parsed from a classical data CSV (classical/data/data_N*_*.csv);
           agent tuples are (id, utility, clearance), so p = 1 - clearance.

Usage
-----
  # Validate: does PPO match the DP optimum on a small instance?
  python -m augmented.rl_train exact  --source random --n 6 --B 2 --G 3 \
         --timesteps 60000 --seed 0
  python -m augmented.rl_train exact  --source csv \
         --csv classical/data/data_N3_d2_B2_G3.csv --timesteps 60000

  # Scale: train the bucketed env beyond the exact-DP wall and beat greedy.
  python -m augmented.rl_train bucket --source random --N 50 --B 2 --G 3 \
         --timesteps 200000 --seed 0

Every run prints the seed and the instances used so results can be reproduced.
"""

import argparse
import ast
import csv
import os
import re

import numpy as np

from augmented.core import mask_from_indices
from augmented.greedy import _myopic_best_pool, greedy_myopic_simulate
from augmented.solver import solve_optimal_dapts
from augmented.rl_env import DaptsBucketEnv, DaptsExactEnv, prior_profile_weights

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "rl_models")


# -------------------------------------------------------------------
# Instance sources
# -------------------------------------------------------------------

def random_instance_generator(n, util_choices=(1.0, 2.0, 3.0),
                            p_low=0.0, p_high=1.0):
    """A generator that draws a fresh random (p, u) on each call.

    Reproducible: the env feeds in its seeded gymnasium Generator.
    """
    util_choices = np.asarray(util_choices, dtype=float)

    def generator(rng):
        p = rng.uniform(p_low, p_high, size=n)
        u = rng.choice(util_choices, size=n)
        return p.tolist(), u.tolist()

    return generator


def fixed_instance_generator(p, u):
    """A generator that always returns the same (p, u) -- ignores the RNG."""
    p, u = list(p), list(u)

    def generator(rng):
        return list(p), list(u)

    return generator


def load_csv_instances(path):
    """Parse a classical data CSV.

    Returns (instances, N, B, G) where instances is a list of (p, u) with
    p = 1 - clearance and u = utility. N, B, G come from the file name.
    """
    base = os.path.basename(path)
    m = re.search(r"N(\d+)_d\d+_B(\d+)_G(\d+)", base)
    if not m:
        raise ValueError(f"cannot parse N, B, G from filename: {base}")
    N, B, G = (int(x) for x in m.groups())

    instances = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            agents = sorted(ast.literal_eval(row["agents"]), key=lambda a: a[0])
            p = [1.0 - float(a[2]) for a in agents]
            u = [float(a[1]) for a in agents]
            instances.append((p, u))
    return instances, N, B, G


def csv_instance_generator(instances):
    """A generator that picks one (p, u) uniformly from a list of instances."""
    def generator(rng):
        p, u = instances[int(rng.integers(len(instances)))]
        return list(p), list(u)

    return generator


# -------------------------------------------------------------------
# Shared rollout (matches DaptsBucketEnv._rollout_reward exactly)
# -------------------------------------------------------------------

def rollout_utility(p, u, B, G, z, first_pool=None):
    """Cleared-utility of: test 1 = first_pool (or greedy), tests 2..B = greedy.

    With first_pool=None this is pure augmented myopic greedy -- the baseline.
    """
    if not first_pool:
        _, _, util = greedy_myopic_simulate(p, u, B, G, z)
        return float(util)

    calls = {"k": 0}

    def selector(pp, uu, GG, nn, cleared):
        first = calls["k"] == 0
        calls["k"] += 1
        if first:
            return first_pool
        return _myopic_best_pool(pp, uu, GG, nn, cleared)

    _, _, util = greedy_myopic_simulate(p, u, B, G, z, pool_selector=selector)
    return float(util)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def train(env, timesteps, seed, model_path=None, verbose=0):
    """Train PPO on a Gymnasium env. Seeded for reproducibility."""
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", env, seed=seed, verbose=verbose)
    model.learn(total_timesteps=timesteps)
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)
        print(f"  model saved to {model_path}.zip")
    return model


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate_exact_vs_dp(model, p, u, B, G):
    """Exact expected utility of the PPO policy vs the DP optimum.

    The PPO policy value is computed exactly: enumerate every latent_state
    profile z, play the policy deterministically, weight by Pr(z).
    """
    n = len(p)
    env = DaptsExactEnv(fixed_instance_generator(p, u), B, G, n)
    weights = prior_profile_weights(p)

    policy_value = 0.0
    for z in range(1 << n):
        if weights[z] == 0.0:
            continue
        obs, _ = env.reset(seed=12345, options={"force_z": z})
        done = False
        episode = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode += reward
        policy_value += weights[z] * episode

    dp_value, _ = solve_optimal_dapts(p, u, B, G)
    return policy_value, dp_value


def evaluate_bucket(model, instance_generator, B, G, N, n_episodes, seed):
    """Mean PPO reward vs mean greedy reward over matched episodes.

    Each episode uses the same instance and same true profile z for both
    PPO (test 1 = PPO pool) and greedy (test 1 = greedy pool).
    """
    env = DaptsBucketEnv(instance_generator, B, G, N)
    ppo_rewards, greedy_rewards = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
        ppo_rewards.append(reward)
        greedy_rewards.append(
            rollout_utility(list(env.p), list(env.u), B, G, env.last_z))
    return float(np.mean(ppo_rewards)), float(np.mean(greedy_rewards))


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def _run_exact(args):
    if args.source == "csv":
        instances, N, B, G = load_csv_instances(args.csv)
        p, u = instances[args.csv_row]
        n = N
        print(f"[exact] CSV {os.path.basename(args.csv)} row {args.csv_row}: "
              f"n={n}, B={B}, G={G}")
    else:
        n, B, G = args.n, args.B, args.G
        rng = np.random.default_rng(args.seed)
        p, u = random_instance_generator(n)(rng)
        print(f"[exact] random instance (seed={args.seed}): n={n}, B={B}, G={G}")

    print(f"  p = {[round(x, 4) for x in p]}")
    print(f"  u = {[round(x, 2) for x in u]}")

    env = DaptsExactEnv(fixed_instance_generator(p, u), B, G, n)
    model_path = os.path.join(args.model_dir, f"exact_n{n}_B{B}_G{G}_s{args.seed}")
    print(f"  training PPO for {args.timesteps} timesteps (seed={args.seed})...")
    model = train(env, args.timesteps, args.seed, model_path,
                  verbose=1 if args.verbose else 0)

    ppo_val, dp_val = evaluate_exact_vs_dp(model, p, u, B, G)
    ratio = ppo_val / dp_val if dp_val else float("nan")
    print("\n--- Validación: PPO vs DP óptimo ---")
    print(f"  PPO (valor exacto de la política) : {ppo_val:.6f}")
    print(f"  DP óptimo                          : {dp_val:.6f}")
    print(f"  cociente PPO / DP                  : {ratio:.4f}")


def _run_bucket(args):
    if args.source == "csv":
        instances, N, B, G = load_csv_instances(args.csv)
        generator = csv_instance_generator(instances)
        print(f"[bucket] CSV {os.path.basename(args.csv)}: "
              f"{len(instances)} instances, N={N}, B={B}, G={G}")
    else:
        N, B, G = args.N, args.B, args.G
        generator = random_instance_generator(N)
        print(f"[bucket] random instances (seed={args.seed}): "
              f"N={N}, B={B}, G={G}")

    env = DaptsBucketEnv(generator, B, G, N)
    model_path = os.path.join(args.model_dir, f"bucket_N{N}_B{B}_G{G}_s{args.seed}")
    print(f"  training PPO for {args.timesteps} timesteps (seed={args.seed})...")
    model = train(env, args.timesteps, args.seed, model_path,
                  verbose=1 if args.verbose else 0)

    ppo_mean, greedy_mean = evaluate_bucket(
        model, generator, B, G, N, args.eval_episodes, args.seed + 10_000)
    print(f"\n--- Evaluación sobre {args.eval_episodes} episodios ---")
    print(f"  PPO    (utilidad media) : {ppo_mean:.4f}")
    print(f"  greedy (utilidad media) : {greedy_mean:.4f}")
    print(f"  diferencia PPO - greedy : {ppo_mean - greedy_mean:+.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="env", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", choices=["random", "csv"], default="random")
    common.add_argument("--csv", help="path to a classical data CSV")
    common.add_argument("--B", type=int, default=2)
    common.add_argument("--G", type=int, default=3)
    common.add_argument("--timesteps", type=int, default=60_000)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    common.add_argument("--verbose", action="store_true")

    pe = sub.add_parser("exact", parents=[common],
                        help="exact belief-MDP, validated against DP")
    pe.add_argument("--n", type=int, default=6)
    pe.add_argument("--csv-row", type=int, default=0)

    pb = sub.add_parser("bucket", parents=[common],
                        help="bucketed env for large N")
    pb.add_argument("--N", type=int, default=50)
    pb.add_argument("--eval-episodes", type=int, default=200)

    args = parser.parse_args(argv)
    if args.source == "csv" and not args.csv:
        parser.error("--source csv requires --csv PATH")

    if args.env == "exact":
        _run_exact(args)
    else:
        _run_bucket(args)


if __name__ == "__main__":
    main()
