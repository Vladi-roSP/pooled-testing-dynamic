"""
VW super-node formulation — formulation check, counterexample, and
empirical comparison on n=6, G=3, B=2.

Setup. After k tests, the covered set S = ∪ pools is built. V = N \\ S.
The VW formulation augments the next-step problem with super-nodes
W = {w_T : T ⊆ S}, each carrying scalar (weight=|T|, prob, util), and
restricting the pool to use at most one w_T. We enumerate over W and
solve for V to pick the next pool.

Two natural readings of prob(w_T):
    (A) all-clear:    prob_A(w_T) = ∏_{i∈T}(1 − p_i)
    (B) any-nonzero_count: prob_B(w_T) = 1 − ∏_{i∈T}(1 − p_i)   (Codex assumption)

Empirical claims demonstrated here:
  1. With prob_A and util_T = Σ u_i, scalar VW EXACTLY reproduces the
     standard myopic-greedy step value (utility-of-pool decomposes
     additively, all-clear probability decomposes multiplicatively).
  2. With prob_B, scalar VW solves a DIFFERENT objective and can pick
     worse pools.
  3. Both scalar variants are blind to the COUNT distribution of r > 0,
     so neither closes the lookahead gap to the DP optimum. We exhibit
     two T's with the same OR-prob but different count distributions —
     any lookahead step will treat them differently.
"""

import math
from itertools import combinations

from augmented.bayesian import (
    _poisson_binomial_pmf,
    bayesian_update_single_test,
)
from augmented.core import (
    indices_from_mask,
    mask_from_indices,
    test_result,
)
from augmented.greedy import (
    greedy_myopic_expected_utility,
    greedy_myopic_simulate,
)
from augmented.solver import solve_optimal_dapts


def _pool_score_myopic(pool_mask, p, u):
    idx = indices_from_mask(pool_mask)
    if not idx:
        return 0.0
    util = sum(u[i] for i in idx)
    prob_clear = 1.0
    for i in idx:
        prob_clear *= 1.0 - p[i]
    return util * prob_clear


def _enumerate_full_pools(S_idx, V_idx, p, u, G):
    universe = S_idx + V_idx
    best_pool, best_val = 0, 0.0
    for size in range(1, G + 1):
        for combo in combinations(universe, size):
            pool = mask_from_indices(combo)
            val = _pool_score_myopic(pool, p, u)
            if val > best_val:
                best_val, best_pool = val, pool
    return best_pool, best_val


def _enumerate_vw(S_idx, V_idx, p, u, G, prob_mode):
    """Enumerate (T ⊆ S, U ⊆ V) with |T|+|U| ≤ G, |t∩W| ≤ 1.

    Score = (util(U) + util(w_T)) · prob_clear(U) · prob(w_T)
    where prob(w_T) is set per `prob_mode` and util(w_T) = Σ_{i∈T} u_i.
    """
    best_pool, best_val = 0, 0.0
    for t_size in range(0, min(len(S_idx), G) + 1):
        for T in combinations(S_idx, t_size):
            if T:
                util_T = sum(u[i] for i in T)
                pclear_T = 1.0
                for i in T:
                    pclear_T *= 1.0 - p[i]
                if prob_mode == "all_clear":
                    prob_T = pclear_T
                elif prob_mode == "or_event":
                    prob_T = 1.0 - pclear_T
                else:
                    raise ValueError(f"unknown prob_mode {prob_mode!r}")
            else:
                util_T, prob_T = 0.0, 1.0

            u_budget = G - t_size
            for u_size in range(0, min(len(V_idx), u_budget) + 1):
                for U in combinations(V_idx, u_size):
                    util_U = sum(u[i] for i in U)
                    pclear_U = 1.0
                    for i in U:
                        pclear_U *= 1.0 - p[i]
                    score = (util_U + util_T) * pclear_U * prob_T
                    if score > best_val:
                        best_val = score
                        best_pool = mask_from_indices(list(U) + list(T))
    return best_pool, best_val


def main():
    n, G, B = 6, 3, 2
    p_prior = [0.10, 0.15, 0.20, 0.08, 0.12, 0.25]
    u = [4.0, 6.0, 3.0, 5.0, 7.0, 4.0]

    print(f"Instance: n={n}, G={G}, B={B}")
    print(f"  prior p = {p_prior}")
    print(f"  util  u = {u}")

    # ----- Step 1 (fixed for reproducibility): test {0, 2, 4} -----
    pool1 = mask_from_indices([0, 2, 4])
    z_true = mask_from_indices([2])
    r1 = test_result(pool1, z_true)
    p_post = bayesian_update_single_test(p_prior, pool1, r1, n)
    S_idx = indices_from_mask(pool1, n)
    V_idx = [i for i in range(n) if i not in S_idx]

    print(f"\nStep 1: pool={S_idx}, observed r={r1}")
    print(f"  posterior p = {[round(x, 4) for x in p_post]}")
    print(f"  S = {S_idx},  V = {V_idx}")

    # ----- Step 2: three formulations of the myopic best-pool problem -----
    pool_full, val_full = _enumerate_full_pools(S_idx, V_idx, p_post, u, G)
    pool_A,    val_A    = _enumerate_vw(S_idx, V_idx, p_post, u, G, "all_clear")
    pool_B,    val_B    = _enumerate_vw(S_idx, V_idx, p_post, u, G, "or_event")

    print("\n--- Step-2 myopic decision: three formulations ---")
    print(f"  full pool enum.   : pool={indices_from_mask(pool_full)}  "
          f"val={val_full:.6f}")
    print(f"  VW  (all-clear)  : pool={indices_from_mask(pool_A)}  "
          f"val={val_A:.6f}")
    print(f"  VW  (or-event)   : pool={indices_from_mask(pool_B)}  "
          f"val={val_B:.6f}")
    print(f"  val(VW-A) == val(full)? {abs(val_A - val_full) < 1e-9}")
    print(f"  val(VW-B) == val(full)? {abs(val_B - val_full) < 1e-9}")

    # ----- DP optimum and true greedy value (end-to-end, B=2) -----
    # greedy_myopic_expected_utility weights branches with the exact
    # P(r | history) at this n (<= EXACT_PMF_MAX_N), so it IS the truthful
    # policy value. The profile-weighted simulation below is kept as an
    # independent cross-check: both numbers must coincide.
    q = [1.0 - pi for pi in p_prior]
    u_greedy_true = 0.0
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p_prior[i] if (z >> i) & 1 else q[i]
        _, _, util_z = greedy_myopic_simulate(p_prior, u, B, G, z)
        u_greedy_true += w * util_z

    u_greedy_recurse = greedy_myopic_expected_utility(p_prior, u, B, G)
    u_opt, F_opt = solve_optimal_dapts(p_prior, u, B, G)

    print("\n--- End-to-end expected utility (B=2) ---")
    print(f"  greedy (true, simulated)     : {u_greedy_true:.6f}")
    print(f"  greedy (recursive, exact pmf): {u_greedy_recurse:.6f}  "
          "(must match the simulated value)")
    print(f"  DP optimum                   : {u_opt:.6f}")
    print(f"  lookahead gap (DP − greedy)  : "
          f"{u_opt - u_greedy_true:.6f}")
    print("  VW scalar formulation reproduces the myopic objective exactly")
    print("  (when prob_A = all-clear), so it cannot shrink this gap on its own.")

    # ----- Counterexample: OR-event compression loses count info -----
    print("\n--- Counterexample: OR compression vs. count distribution ---")
    p_a = 0.5
    p_bc = 1 - 0.5 ** 0.5
    or1 = 1 - (1 - p_a)
    or2 = 1 - (1 - p_bc) ** 2
    pmf1 = _poisson_binomial_pmf([p_a])
    pmf2 = _poisson_binomial_pmf([p_bc, p_bc])
    H1 = -sum(pi * math.log2(pi) for pi in pmf1 if pi > 0)
    H2 = -sum(pi * math.log2(pi) for pi in pmf2 if pi > 0)
    print(f"  T1 = {{a}}      p_a   = {p_a:.4f}")
    print(f"  T2 = {{b, c}}   p_b=p_c = {p_bc:.4f}")
    print(f"  OR-prob       :  T1 = {or1:.4f},  T2 = {or2:.4f}  (EQUAL)")
    print(f"  all-clear prob:  T1 = {1-or1:.4f},  T2 = {1-or2:.4f}  (EQUAL)")
    print(f"  count PMF T1  : {[round(x, 4) for x in pmf1]}")
    print(f"  count PMF T2  : {[round(x, 4) for x in pmf2]}")
    print(f"  Shannon H(r)  :  T1 = {H1:.4f} bits,  T2 = {H2:.4f} bits  "
          f"(diff = {H2-H1:+.4f})")
    print("  ⇒ T2 carries strictly more information about the joint Z than T1,")
    print("    even though *both* scalar summaries (OR or all-clear) coincide.")
    print("    Any scalar (weight, prob, util) encoding of w_T cannot")
    print("    distinguish them; only the full count PMF can.")

    # ----- Scaling: how does the lookahead gap grow with B? -----
    print("\n--- Lookahead gap as a function of B (n=6, G=3) ---")
    print("  B   greedy(true)   DP optimum   gap")
    for B_eval in (2, 3):
        u_g = 0.0
        q = [1.0 - pi for pi in p_prior]
        for z in range(1 << n):
            w = 1.0
            for i in range(n):
                w *= p_prior[i] if (z >> i) & 1 else q[i]
            _, _, util_z = greedy_myopic_simulate(p_prior, u, B_eval, G, z)
            u_g += w * util_z
        u_dp, _ = solve_optimal_dapts(p_prior, u, B_eval, G)
        print(f"  {B_eval}   {u_g:11.6f}   {u_dp:10.6f}   {u_dp-u_g:+.6f}")
    print("  VW scalar formulation is myopic by construction → it lives on")
    print("  the 'greedy(true)' line. To close the gap, VW must carry the")
    print("  full count PMF of w_T (i.e., a non-scalar super-node), at which")
    print("  point it becomes equivalent to direct DP enumeration.")


if __name__ == "__main__":
    main()
