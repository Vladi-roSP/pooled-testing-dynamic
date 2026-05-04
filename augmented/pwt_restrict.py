"""
PWT super-node restriction (Q3 from docs/notes/pwt_submodularity.md).

Question. The PWT framework enumerates W = {w_T : T ⊆ S, T ≠ ∅}, which is
2^|S|−1 super-nodes. Can we restrict to a top-L subset using a cheap
ranking heuristic, without losing the optimal pool?

Approach. For each candidate T, the "real" value is
    val(T) = max_{U ⊆ V, |U| ≤ G−|T|}  (Σ u in T∪U) · ∏_{i ∈ T∪U} (1 − p_i)
Computing val(T) is exactly the work we wanted to save. We measure
L_min(rank) = the smallest L such that the top-L T's by `rank` contain
an optimal one — i.e., a T* with val(T*) = max_T val(T).

Heuristics tested (each O(|T|)):
    self_score(T) = (Σ u_T) · ∏(1 − p_T)      myopic value of pool=T alone
    prob(T)       = ∏(1 − p_T)                 all-clear prob alone
    util(T)       = Σ u_T                      utility sum alone
    random                                      sanity baseline

All heuristics keep the V-only pool (T = ∅) implicitly available; we
only rank non-empty T's.

Empirical setup. Random p (low-prevalence) and u, simulate two
augmented tests to build S, then evaluate the restriction at step 3.
"""

import math
import random
from itertools import combinations

from augmented.bayesian import (
    _poisson_binomial_pmf,
    bayesian_update_single_test,
)
from augmented.core import (
    indices_from_mask,
    mask_from_indices,
)


def _prod(xs):
    out = 1.0
    for x in xs:
        out *= x
    return out


def _pool_value(pool_idx, p, u):
    if not pool_idx:
        return 0.0
    util = sum(u[i] for i in pool_idx)
    pclear = _prod((1 - p[i]) for i in pool_idx)
    return util * pclear


def _best_pool_with_T(T, V_idx, p, u, G):
    """Best pool t = T ∪ U with U ⊆ V, |t| ≤ G."""
    t_size = len(T)
    if t_size > G:
        return 0.0
    u_budget = G - t_size
    best_val = 0.0
    u_min_size = 1 if not T else 0  # require non-empty pool overall
    for u_size in range(u_min_size, min(len(V_idx), u_budget) + 1):
        for U in combinations(V_idx, u_size):
            val = _pool_value(list(T) + list(U), p, u)
            if val > best_val:
                best_val = val
    return best_val


def _entropy(pmf):
    return -sum(pi * math.log2(pi) for pi in pmf if pi > 0)


def _best_v_partner_per_size(V_idx, p, u, G):
    """Precompute, for each k = 0..G, the best V-pool of size exactly k by
    its myopic value, returning (util_sum, prob_clear, val).

    For k > |V|, returns the best of size |V|. The lookup gives a tight
    surrogate for "what U could partner with a T of size G−|T|"."""
    out = {0: (0.0, 1.0, 0.0)}
    for k in range(1, G + 1):
        best = (0.0, 0.0, 0.0)
        if k <= len(V_idx):
            for U in combinations(V_idx, k):
                util = sum(u[i] for i in U)
                pclear = _prod((1 - p[i]) for i in U)
                val = util * pclear
                if val > best[2]:
                    best = (util, pclear, val)
        else:
            best = out[len(V_idx)]
        out[k] = best
    return out


def restriction_experiment(p_post, S_idx, V_idx, u, G, rand_seed=0,
                           lambdas=(1.0, 5.0, 25.0)):
    """Compute L_min for each ranking heuristic on a fixed (p, S, V, u, G).

    Heuristics evaluated:
      self     : (Σu_T) · ∏(1−p_T)            myopic value of pool=T alone
      prob     : ∏(1−p_T)                      all-clear prob
      util     : Σu_T                          utility sum
      ent_λ    : self + λ · H(r_T)             entropy-augmented (one per λ)
      partner  : (Σu_T + u*) · ∏(1−p_T) · p*  best V-partner of size G−|T|
      rand     : uniform random                sanity baseline
    """
    # All non-empty T ⊆ S with |T| ≤ G
    all_T = []
    for size in range(1, min(len(S_idx), G) + 1):
        for combo in combinations(S_idx, size):
            all_T.append(combo)

    # Ground truth values
    val_empty = _best_pool_with_T((), V_idx, p_post, u, G)
    val_T = {T: _best_pool_with_T(T, V_idx, p_post, u, G) for T in all_T}
    val_full = max([val_empty] + list(val_T.values()))

    # Precompute best V partner per size (used by h_partner)
    best_partner = _best_v_partner_per_size(V_idx, p_post, u, G)

    # Per-T stats used by multiple heuristics
    T_self = {}
    T_entropy = {}
    for T in all_T:
        util_T = sum(u[i] for i in T)
        prob_T = _prod((1 - p_post[i]) for i in T)
        T_self[T] = util_T * prob_T
        pmf = _poisson_binomial_pmf([p_post[i] for i in T])
        T_entropy[T] = _entropy(pmf)

    # Heuristics
    def h_self(T):
        return T_self[T]

    def h_prob(T):
        return _prod((1 - p_post[i]) for i in T)

    def h_util(T):
        return sum(u[i] for i in T)

    def h_partner(T):
        util_T = sum(u[i] for i in T)
        prob_T = _prod((1 - p_post[i]) for i in T)
        u_p, p_p, _ = best_partner[G - len(T)]
        return (util_T + u_p) * prob_T * p_p

    def make_h_entropy(lam):
        def h(T):
            return T_self[T] + lam * T_entropy[T]
        return h

    rng = random.Random(rand_seed)
    rand_keys = {T: rng.random() for T in all_T}

    def h_rand(T):
        return rand_keys[T]

    def L_min(rank_fn):
        if val_empty >= val_full - 1e-12:
            return 0
        ranked = sorted(all_T, key=lambda T: -rank_fn(T))
        best_so_far = val_empty
        for L, T in enumerate(ranked, 1):
            if val_T[T] > best_so_far:
                best_so_far = val_T[T]
            if best_so_far >= val_full - 1e-12:
                return L
        return len(ranked)

    out = {
        "W_size": len(all_T),
        "val_full": val_full,
        "val_empty": val_empty,
        "L_min_self": L_min(h_self),
        "L_min_prob": L_min(h_prob),
        "L_min_util": L_min(h_util),
        "L_min_partner": L_min(h_partner),
        "L_min_rand": L_min(h_rand),
    }
    for lam in lambdas:
        out[f"L_min_ent_{lam:g}"] = L_min(make_h_entropy(lam))
    return out


def _sample_r(pool_idx, p, rng):
    return sum(1 for i in pool_idx if rng.random() < p[i])


def adversarial_instance():
    """Hand-crafted instance where self_score's L_min is large.

    Mechanism: include in S a high-(util · prob_clear) BIG-T candidate
    whose val(T) is mediocre because the prob-clear factor multiplies
    multiplicatively with V's prob-clear, sinking the joint pool. The
    truly optimal T is a *singleton* whose self_score is lower but whose
    val(T) is higher because |T|=1 leaves more budget for V.

    Returns (p_post, S_idx, V_idx, u, G).
    """
    n = 6
    G = 4
    # S = {0, 1, 2, 3}, V = {4, 5}
    p_post = [0.01, 0.01, 0.60, 0.05, 0.05, 0.50]
    u = [1.0, 1.0, 100.0, 50.0, 60.0, 80.0]
    S_idx = [0, 1, 2, 3]
    V_idx = [4, 5]
    return p_post, S_idx, V_idx, u, G


def run_trial(p_prior, u, n, G, k_priors, trial_seed):
    """Run k_priors random tests on (p_prior, u), then evaluate restriction.

    Returns the experiment dict, or None if S ends up empty / |T| > G is
    forced (degenerate).
    """
    rng = random.Random(trial_seed)
    p_cur = list(p_prior)
    S = 0
    for _ in range(k_priors):
        pool_idx = tuple(sorted(rng.sample(range(n), G)))
        pool_mask = mask_from_indices(pool_idx)
        r = _sample_r(pool_idx, p_cur, rng)
        p_cur = bayesian_update_single_test(p_cur, pool_mask, r, n)
        S |= pool_mask
    S_idx = indices_from_mask(S)
    V_idx = [i for i in range(n) if i not in S_idx]
    if not S_idx:
        return None
    return restriction_experiment(p_cur, S_idx, V_idx, u, G,
                                  rand_seed=trial_seed)


def main():
    n, G = 10, 4
    K = 20

    # Reproducible prior + utilities
    rng_inst = random.Random(7)
    p_prior = [rng_inst.uniform(0.05, 0.35) for _ in range(n)]
    u = [rng_inst.uniform(1.0, 10.0) for _ in range(n)]

    print(f"Setup: n={n}, G={G}, K={K} random 2-step histories")
    print(f"  p_prior = {[round(x, 3) for x in p_prior]}")
    print(f"  u       = {[round(x, 2) for x in u]}\n")

    results = []
    for trial in range(K):
        rng = random.Random(1000 + trial)
        # Two random pools of size G; observe r's via Bernoulli draws on prior
        pool1_idx = tuple(sorted(rng.sample(range(n), G)))
        pool1 = mask_from_indices(pool1_idx)
        r1 = _sample_r(pool1_idx, p_prior, rng)
        p1 = bayesian_update_single_test(p_prior, pool1, r1, n)

        pool2_idx = tuple(sorted(rng.sample(range(n), G)))
        pool2 = mask_from_indices(pool2_idx)
        r2 = _sample_r(pool2_idx, p1, rng)
        p2 = bayesian_update_single_test(p1, pool2, r2, n)

        S = pool1 | pool2
        S_idx = indices_from_mask(S)
        V_idx = [i for i in range(n) if i not in S_idx]
        if not S_idx:
            continue

        res = restriction_experiment(p2, S_idx, V_idx, u, G, rand_seed=trial)
        results.append(res)

    keys = sorted({k for k in results[0] if k.startswith("L_min_")})
    K_eff = len(results)
    means = {k: sum(r[k] for r in results) / K_eff for k in keys}
    maxes = {k: max(r[k] for r in results) for k in keys}
    mean_W = sum(r["W_size"] for r in results) / K_eff
    print(f"Summary over {K_eff} trials:    mean |W| = {mean_W:.1f}")
    print(f"  heuristic              mean L_min   max L_min   "
          f"mean / |W|")
    for k in keys:
        label = k.replace("L_min_", "")
        ratio = means[k] / mean_W if mean_W > 0 else 0.0
        print(f"  {label:<20}   {means[k]:>10.2f}   {maxes[k]:>9}   "
              f"{ratio:>10.3f}")

    # Hand-crafted adversarial
    print("\n--- Adversarial instance (self_score worst-case demo) ---")
    p_adv, S_adv, V_adv, u_adv, G_adv = adversarial_instance()
    print(f"  S = {S_adv}, V = {V_adv}, G = {G_adv}")
    print(f"  p = {p_adv}")
    print(f"  u = {u_adv}")
    res_adv = restriction_experiment(p_adv, S_adv, V_adv, u_adv, G_adv,
                                     rand_seed=0)
    keys_adv = sorted([k for k in res_adv if k.startswith("L_min_")])
    print(f"  |W| = {res_adv['W_size']}    val_full = {res_adv['val_full']:.4f}"
          f"    val_empty = {res_adv['val_empty']:.4f}")
    print(f"  heuristic              L_min   L_min / |W|")
    for k in keys_adv:
        label = k.replace("L_min_", "")
        ratio = res_adv[k] / res_adv['W_size'] if res_adv['W_size'] else 0.0
        print(f"  {label:<20}   {res_adv[k]:>5}   {ratio:>10.3f}")


if __name__ == "__main__":
    main()
