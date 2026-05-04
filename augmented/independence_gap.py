"""
Independence-heuristic gap: how close is the true joint posterior of a pool's
outcome to the product-of-marginals approximation used by the greedy scoring?

The greedy strategies approximate the joint posterior of (Z_i)_{i in t} given
the test history H as the product of the marginals tilde_p_i = P(Z_i=1 | H).
Under that assumption, the PMF of r_t = |t ∩ Z| is a Poisson-Binomial on
(tilde_p_i). In reality, tests induce dependencies between indicators, so
the true joint PMF can differ from the Poisson-Binomial one.

Concrete case where the gap is large: if t' ⊂ t was tested with
0 < r' < |t'| (intermediate count), then the members of t' become
correlated under the posterior without becoming degenerate (the
marginals tilde_p_i stay strictly in (0, 1)). Then the joint PMF of
r_t has P(r_t = 0 | H) = 0 while the Poisson-Binomial heuristic puts
positive mass there. The cleanest instance (used in the unit tests
and the worked example) is a symmetric prior with t' = {0, 1} and
r' = 1, which forces tilde_p_0 = tilde_p_1 = 0.5.

Cases where the marginals already capture the dependency and the
heuristic is exact:
  * t' = t with r' = 0 or r' = |t'| (fully determined by one test).
  * t does not overlap the history at all (conditional independence
    holds trivially).

This module provides:
  - exact_pool_pmf: the true posterior PMF of r_t by enumerating worlds.
  - independence_pool_pmf: Poisson-Binomial PMF from posterior marginals.
  - gap_summary: TV distance and endpoint-by-endpoint comparisons.
  - run_experiment: sweep random priors and histories, aggregate gaps.
"""

import random
from statistics import mean, median

from augmented.core import (
    indices_from_mask, popcount, test_result, all_pools, all_pools_from_mask,
)
from augmented.bayesian import (
    bayesian_update_by_counting,
    _poisson_binomial_pmf,
)
from augmented.greedy import greedy_myopic_simulate


# -------------------------------------------------------------------
# Core quantities
# -------------------------------------------------------------------

def exact_pool_pmf(p, history, pool_mask, n):
    """Exact posterior PMF of r_t = |t ∩ Z| given history H.

    Enumerates all 2^n profiles, restricts to those consistent with H,
    and aggregates prior weights by the outcome r_t they produce.

    Returns a list of length popcount(pool_mask)+1 where entry k is
    P(r_t = k | H). If no profile is consistent with the history, the
    returned PMF has total mass 0 (degenerate; caller can skip).
    """
    q = [1.0 - pi for pi in p]
    m = popcount(pool_mask)
    pmf = [0.0] * (m + 1)
    total = 0.0

    for z in range(1 << n):
        ok = True
        for t, r in history:
            if test_result(t, z) != r:
                ok = False
                break
        if not ok:
            continue

        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else q[i]

        pmf[test_result(pool_mask, z)] += w
        total += w

    if total > 0:
        pmf = [v / total for v in pmf]
    return pmf


def independence_pool_pmf(p, history, pool_mask, n, marginals=None):
    """Product-of-marginals PMF for r_t.

    Uses posterior marginals tilde_p_i = P(Z_i=1 | H) (computed exactly
    by counting) and returns the Poisson-Binomial PMF on (tilde_p_i) for
    i in pool. Pass precomputed `marginals` to avoid recomputing.
    """
    if marginals is None:
        marginals = bayesian_update_by_counting(p, history, n)
    pool_idx = indices_from_mask(pool_mask, n)
    return _poisson_binomial_pmf([marginals[i] for i in pool_idx])


def tv_distance(a, b):
    """Total variation distance between two PMFs of equal length."""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def gap_summary(p, history, pool_mask, n, marginals=None):
    """Compare exact and independence-heuristic PMFs for one pool.

    Returns a dict with both PMFs and scalar gap metrics. The endpoints
    are the two quantities with direct operational meaning:
      * r=0 ("clear"): drives the greedy myopic scoring.
      * r=|t| ("all infected"): the literal reading of the user's formula.
    """
    exact = exact_pool_pmf(p, history, pool_mask, n)
    heur = independence_pool_pmf(p, history, pool_mask, n, marginals=marginals)
    m = len(exact) - 1
    return {
        'pool_size': m,
        'exact_pmf': exact,
        'heuristic_pmf': heur,
        'tv': tv_distance(exact, heur),
        'gap_r0': heur[0] - exact[0],
        'gap_rmax': heur[m] - exact[m],
        'abs_gap_r0': abs(heur[0] - exact[0]),
        'abs_gap_rmax': abs(heur[m] - exact[m]),
    }


# -------------------------------------------------------------------
# Experiment driver
# -------------------------------------------------------------------

def _sample_prior(n, rng, low=0.05, high=0.5):
    return [rng.uniform(low, high) for _ in range(n)]


def _sample_profile(p, rng):
    z = 0
    for i, pi in enumerate(p):
        if rng.random() < pi:
            z |= 1 << i
    return z


def run_experiment(n=8, B=3, G=3, num_instances=200, pool_sizes=None,
                   prior_low=0.05, prior_high=0.5, seed=0,
                   history_strategy='greedy'):
    """Sweep random (prior, ground truth) pairs, simulate histories,
    and record gap metrics for every candidate pool on each instance.

    Parameters
    ----------
    n : int
        Population size (kept small: O(2^n) enumeration).
    B : int
        Number of tests per history.
    G : int
        Max pool size used when generating histories and candidate pools.
    num_instances : int
        Number of (prior, ground truth) draws.
    pool_sizes : iterable[int] or None
        Which candidate pool sizes to score. Defaults to 2..G.
    history_strategy : 'greedy' | 'random' | 'none'
        How to generate the conditioning history.

    Returns
    -------
    list[dict]
        One row per (instance, candidate pool). Fields include the gap
        summary plus the parameters that generated the row.
    """
    rng = random.Random(seed)
    if pool_sizes is None:
        pool_sizes = list(range(2, G + 1))

    u = [1.0] * n  # utilities only matter for greedy selection
    rows = []

    for inst in range(num_instances):
        p = _sample_prior(n, rng, prior_low, prior_high)
        z_mask = _sample_profile(p, rng)

        if history_strategy == 'greedy':
            history, _, _ = greedy_myopic_simulate(p, u, B, G, z_mask)
        elif history_strategy == 'random':
            history = ()
            for _ in range(B):
                size = rng.randint(1, G)
                members = rng.sample(range(n), size)
                pm = 0
                for i in members:
                    pm |= 1 << i
                history = history + ((pm, test_result(pm, z_mask)),)
        elif history_strategy == 'none':
            history = ()
        else:
            raise ValueError(f"unknown history_strategy={history_strategy!r}")

        marginals = bayesian_update_by_counting(p, history, n)

        # Candidate pools: all subsets of given sizes over the full
        # population. Not restricted to "active" mask, so we also see
        # overlaps with cleared/confirmed individuals.
        full_mask = (1 << n) - 1
        candidates = [t for t in all_pools_from_mask(full_mask, G,
                                                     include_empty=False)
                      if popcount(t) in pool_sizes]

        for t in candidates:
            summary = gap_summary(p, history, t, n, marginals=marginals)
            summary.update({
                'instance': inst,
                'pool_mask': t,
                'history_len': len(history),
                'history_strategy': history_strategy,
                'prior_mean': sum(p) / n,
                'num_infected_truth': popcount(z_mask),
            })
            rows.append(summary)

    return rows


# -------------------------------------------------------------------
# "The payoff": greedy with exact P(r=0 | H) scoring
# -------------------------------------------------------------------
#
# The myopic greedy currently scores a pool t by
#     score(t) = prod_{i in t} (1 - tilde_p_i) * sum_{i in t} u_i,
# using the independence heuristic. When the exact joint posterior is
# far from the product-of-marginals, this can mislead the selection.
# The functions below score pools using the EXACT conditional
# probability P(r_t = 0 | H), computed from the set of infection
# profiles still consistent with the history.

def _prior_weights_indep(p, n):
    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n
    w = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz
    return w


def _exact_best_pool(remaining, cleared, u, pools, w, n):
    """Pick the pool that maximizes exact P(r=0 | H) * (new utility)."""
    total_mass = sum(w[z] for z in remaining)
    if total_mass == 0:
        return 0, 0.0

    best_pool, best_score = 0, -1.0
    for pool in pools:
        mass_zero = sum(w[z] for z in remaining
                        if test_result(pool, z) == 0)
        prob_clear = mass_zero / total_mass
        gain = sum(u[i] for i in range(n)
                   if (pool >> i & 1) and not (cleared >> i & 1))
        score = prob_clear * gain
        if score > best_score:
            best_score, best_pool = score, pool
    return best_pool, best_score


def exact_greedy_myopic_expected_utility(p, u, B, G):
    """Expected utility of myopic greedy that uses EXACT P(r=0|H).

    Same interface as greedy.greedy_myopic_expected_utility, but the
    per-step pool choice is scored with the exact conditional
    probability of clearing (read off from the remaining consistent
    profiles) rather than the product-of-marginals heuristic.

    Note: the recursion here keeps a `remaining` frozenset in the
    state, so it runs in O(2^n * pool-count * B * reachable-states).
    Intended for small n <= 8; for larger n use Gibbs-based estimates.
    """
    n = len(p)
    w = _prior_weights_indep(p, n)
    pools = all_pools(n, G, include_empty=False)
    all_z = frozenset(range(1 << n))
    memo = {}

    def recurse(k, remaining, cleared):
        key = (k, remaining, cleared)
        if key in memo:
            return memo[key]

        if k == B or not remaining:
            util = sum(u[i] for i in range(n) if (cleared >> i & 1))
            memo[key] = util
            return util

        pool, score = _exact_best_pool(remaining, cleared, u, pools, w, n)
        if pool == 0 or score <= 0.0:
            util = sum(u[i] for i in range(n) if (cleared >> i & 1))
            memo[key] = util
            return util

        # Expected value over outcomes r
        total_mass = sum(w[z] for z in remaining)
        buckets = {}
        for z in remaining:
            r = test_result(pool, z)
            buckets.setdefault(r, []).append(z)

        ev = 0.0
        for r, zs in buckets.items():
            mass_r = sum(w[z] for z in zs)
            prob_r = mass_r / total_mass
            new_cleared = cleared | pool if r == 0 else cleared
            ev += prob_r * recurse(k + 1, frozenset(zs), new_cleared)

        memo[key] = ev
        return ev

    return recurse(0, all_z, 0)


def aggregate(rows):
    """Compact summary stats stratified by pool size."""
    by_size = {}
    for row in rows:
        by_size.setdefault(row['pool_size'], []).append(row)

    out = {}
    for size, group in sorted(by_size.items()):
        tvs = [r['tv'] for r in group]
        r0 = [r['abs_gap_r0'] for r in group]
        rmax = [r['abs_gap_rmax'] for r in group]
        out[size] = {
            'count': len(group),
            'tv_mean': mean(tvs),
            'tv_median': median(tvs),
            'tv_max': max(tvs),
            'tv_p95': sorted(tvs)[int(0.95 * (len(tvs) - 1))],
            'abs_gap_r0_mean': mean(r0),
            'abs_gap_r0_max': max(r0),
            'abs_gap_rmax_mean': mean(rmax),
            'abs_gap_rmax_max': max(rmax),
        }
    return out
