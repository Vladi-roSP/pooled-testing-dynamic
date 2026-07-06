"""
Sweep VW super-node restriction across multiple regimes (Q3 robustness).

For each regime, draws random priors and utilities of the specified
shape, runs `run_trial` K times, and reports L_min stats per heuristic.
The goal is to see whether the self_score heuristic's tight L_min holds
across regimes (low/high prevalence, larger n, deeper history,
heterogeneous utilities, bimodal prevalence).
"""

import random
import statistics

from augmented.vw_restrict import run_trial


def _gen_population(n, p_dist, u_dist, rng):
    """Build (p_prior, u) according to distribution specs."""
    if p_dist[0] == "uniform":
        lo, hi = p_dist[1], p_dist[2]
        p_prior = [rng.uniform(lo, hi) for _ in range(n)]
    elif p_dist[0] == "bimodal":
        lo_p, hi_p, frac_hi = p_dist[1], p_dist[2], p_dist[3]
        p_prior = [hi_p if rng.random() < frac_hi else lo_p for _ in range(n)]
    else:
        raise ValueError(p_dist[0])

    if u_dist[0] == "uniform":
        lo, hi = u_dist[1], u_dist[2]
        u = [rng.uniform(lo, hi) for _ in range(n)]
    elif u_dist[0] == "outlier":
        u_outlier, lo, hi = u_dist[1], u_dist[2], u_dist[3]
        u = [rng.uniform(lo, hi) for _ in range(n)]
        u[0] = u_outlier
    else:
        raise ValueError(u_dist[0])

    return p_prior, u


def run_regime(name, n, G, k_priors, p_dist, u_dist, K=20, base_seed=0):
    """Run K trials of one regime (each with its own p, u) and aggregate."""
    results = []
    for trial in range(K):
        rng = random.Random(base_seed + 10_000 + trial)
        p_prior, u = _gen_population(n, p_dist, u_dist, rng)
        # Use a fresh seed for the trial's drawing, deterministic per trial
        res = run_trial(p_prior, u, n, G, k_priors,
                        trial_seed=base_seed + 1000 + trial)
        if res is not None:
            results.append(res)

    K_eff = len(results)
    if K_eff == 0:
        return None

    keys = sorted({k for k in results[0] if k.startswith("L_min_")})
    means = {k: statistics.mean(r[k] for r in results) for k in keys}
    maxes = {k: max(r[k] for r in results) for k in keys}
    mean_W = statistics.mean(r["W_size"] for r in results)

    return {
        "name": name,
        "K_eff": K_eff,
        "mean_W": mean_W,
        "means": means,
        "maxes": maxes,
        "ratios": {k: means[k] / mean_W if mean_W > 0 else 0.0 for k in keys},
    }


REGIMES = [
    # name, n, G, k_priors, p_dist, u_dist
    ("baseline (n=10,G=4,2-step,low p)", 10, 4, 2,
     ("uniform", 0.05, 0.35), ("uniform", 1.0, 10.0)),
    ("high prevalence (n=10,G=4,p~U[0.4,0.7])", 10, 4, 2,
     ("uniform", 0.40, 0.70), ("uniform", 1.0, 10.0)),
    ("larger n (n=15,G=5,2-step,low p)", 15, 5, 2,
     ("uniform", 0.05, 0.35), ("uniform", 1.0, 10.0)),
    ("deep history (n=10,G=3,3-step)", 10, 3, 3,
     ("uniform", 0.05, 0.35), ("uniform", 1.0, 10.0)),
    ("bimodal p (10/90 mix of 0.05/0.7)", 10, 4, 2,
     ("bimodal", 0.05, 0.70, 0.10), ("uniform", 1.0, 10.0)),
    ("outlier utility (u_0=50, rest U[1,10])", 10, 4, 2,
     ("uniform", 0.05, 0.35), ("outlier", 50.0, 1.0, 10.0)),
]


def main():
    cols = ["partner", "self", "ent_1", "prob", "util", "rand"]
    print(f"{'regime':<40}  {'K':>3} {'|W|':>6}    " +
          "  ".join(f"{c:>9}" for c in cols))
    print(f"{'-' * 40}  {'-' * 3} {'-' * 6}    " +
          "  ".join("-" * 9 for _ in cols))

    rows = []
    for spec in REGIMES:
        name, n, G, k, pd, ud = spec
        out = run_regime(name, n, G, k, pd, ud, K=20, base_seed=42)
        if out is None:
            print(f"{name:<40}  (degenerate)")
            continue
        rows.append(out)
        m, x = out["means"], out["maxes"]
        cells = []
        for c in cols:
            mk = f"L_min_{c}"
            cells.append(f"{m[mk]:>4.1f}/{x[mk]:<3}")
        print(f"{name:<40}  {out['K_eff']:>3} {out['mean_W']:>6.1f}    " +
              "  ".join(cells))

    print("\nFormat: mean / max   |W| = number of non-empty T's   K = trials")
    print("Heuristic legend:")
    print("  partner = (Σu_T + u*) · ∏(1−p_T) · p*      [u*, p* = best V-pool of size G−|T|]")
    print("  self    = (Σu_T) · ∏(1−p_T)                 [pool=T alone]")
    print("  ent_λ   = self + λ · H(r_T)                 [entropy of count PMF]")
    print("  prob    = ∏(1−p_T)                          [all-clear prob]")
    print("  util    = Σu_T                              [utility sum]")
    print("  rand    = uniform                           [sanity baseline]")


if __name__ == "__main__":
    main()
