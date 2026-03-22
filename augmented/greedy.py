"""
Greedy algorithms for augmented pooled testing.

Three greedy strategies:
  1. greedy_myopic:  at each step, pick pool maximizing P(r=0) * Σ u_i
     with sequential single-test Bayesian updates.
  2. greedy_lookahead: at each step, pick pool maximizing expected utility
     over ALL possible outcomes (r=0,1,...,|t|) with Bayesian updates.
  3. greedy_myopic_counting: same as myopic but uses full-history
     Bayesian update by counting over all consistent worlds.

Key insight: the myopic pool selection is identical for classical and
augmented tests (only r=0 matters for immediate utility). The augmented
information helps in FUTURE steps through better Bayesian posteriors.
"""

from augmented.core import (all_pools, all_pools_from_mask, compute_active_mask,
                            indices_from_mask, test_result, mask_str)
from augmented.bayesian import (bayesian_update_single_test,
                                bayesian_update_by_counting,
                                gibbs_update,
                                _poisson_binomial_pmf)


# -------------------------------------------------------------------
# Myopic greedy: maximize immediate expected gain
# -------------------------------------------------------------------

def _myopic_best_pool(p, u, G, n, cleared_mask, use_filtering=True):
    """Pick pool maximizing P(r=0) * Σ u_i for uncleared members.

    Score(t) = prod(1-p_i for i in t) * sum(u_i for i in t if i not cleared)

    If use_filtering=True, only considers pools from active individuals
    (those not yet cleared and not confirmed infected).
    """
    if use_filtering:
        active_mask, _ = compute_active_mask(p, cleared_mask, n)
        if active_mask == 0:
            return 0  # no uncertain individuals left
        pools = all_pools_from_mask(active_mask, G, include_empty=False)
    else:
        pools = all_pools(n, G, include_empty=False)

    best_pool, best_score = 0, 0.0

    for pool in pools:
        pool_idx = indices_from_mask(pool, n)
        prob_clear = 1.0
        for i in pool_idx:
            prob_clear *= (1.0 - p[i])

        gain = sum(u[i] for i in pool_idx if not (cleared_mask >> i & 1))
        score = prob_clear * gain

        if score > best_score:
            best_score = score
            best_pool = pool

    return best_pool


def greedy_myopic_simulate(p, u, B, G, z_mask, pool_selector=None):
    """Simulate myopic greedy on a fixed infection profile z_mask.

    At each step: pick best myopic pool, observe augmented result,
    update beliefs via Bayesian update, repeat.

    Parameters
    ----------
    pool_selector : callable or None
        If provided, called as pool_selector(p, u, G, n, cleared_mask)
        to select the pool at each step.  Defaults to _myopic_best_pool.

    Returns (history, cleared_mask, utility).
    """
    n = len(p)
    current_p = list(p)
    cleared_mask = 0
    history = ()

    for _ in range(B):
        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            break  # no useful pool
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool
        current_p = bayesian_update_single_test(current_p, pool, r, n)

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


def greedy_myopic_expected_utility(p, u, B, G, pool_selector=None):
    """Expected utility of the myopic greedy strategy.

    At each step picks the myopic-best pool, then recurses over all
    possible outcomes (r=0,...,|t|) weighted by their probabilities.

    Parameters
    ----------
    pool_selector : callable or None
        If provided, called as pool_selector(p, u, G, n, cleared_mask)
        to select the pool at each step.  Defaults to _myopic_best_pool.
    """
    n = len(p)

    def recurse(current_p, b, cleared_mask):
        if b == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        pool_idx = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([current_p[i] for i in pool_idx])

        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(current_p, pool, r, n)
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            ev += pmf[r] * recurse(new_p, b - 1, new_cleared)
        return ev

    return recurse(list(p), B, 0)


# -------------------------------------------------------------------
# Lookahead greedy: maximize total expected utility (not just immediate)
# -------------------------------------------------------------------

def _lookahead_best_pool(p, u, G, n, b, cleared_mask):
    """Pick pool maximizing E[total utility over remaining b tests].

    For each candidate pool, sums over all possible outcomes r=0,...,|t|,
    applies Bayesian update, and recursively evaluates future utility.
    Returns (best_pool, expected_utility).
    """
    if b == 0:
        return 0, sum(u[i] for i in indices_from_mask(cleared_mask, n))

    pools = all_pools(n, G, include_empty=False)
    best_pool, best_ev = 0, -1.0

    for pool in pools:
        pool_idx = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([p[i] for i in pool_idx])

        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(p, pool, r, n)
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            # Recurse: for future steps, use MYOPIC selection (not full lookahead)
            # Full lookahead at every level would be the optimal DP (too expensive).
            _, future_val = _greedy_future(new_p, u, G, n, b - 1, new_cleared)
            ev += pmf[r] * future_val

        if ev > best_ev:
            best_ev = ev
            best_pool = pool

    # Also consider doing nothing
    nothing_val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    if b > 0:
        _, future_nothing = _greedy_future(p, u, G, n, b - 1, cleared_mask)
        nothing_val = future_nothing
    if nothing_val > best_ev:
        best_ev = nothing_val
        best_pool = 0

    return best_pool, best_ev


def _greedy_future(p, u, G, n, b, cleared_mask):
    """Evaluate expected utility of myopic greedy for remaining b tests."""
    if b == 0:
        val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return 0, val

    pool = _myopic_best_pool(p, u, G, n, cleared_mask)
    if pool == 0:
        val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return 0, val

    pool_idx = indices_from_mask(pool, n)
    pmf = _poisson_binomial_pmf([p[i] for i in pool_idx])

    ev = 0.0
    for r in range(len(pool_idx) + 1):
        if pmf[r] < 1e-15:
            continue
        new_p = bayesian_update_single_test(p, pool, r, n)
        new_cleared = cleared_mask | pool if r == 0 else cleared_mask
        _, future_val = _greedy_future(new_p, u, G, n, b - 1, new_cleared)
        ev += pmf[r] * future_val

    return pool, ev


def greedy_lookahead_simulate(p, u, B, G, z_mask):
    """Simulate lookahead greedy on a fixed infection profile.

    At step 1: uses full lookahead to pick the best pool.
    At steps 2+: falls back to myopic (otherwise it's the full DP solver).

    Returns (history, cleared_mask, utility).
    """
    n = len(p)
    current_p = list(p)
    cleared_mask = 0
    history = ()

    for step in range(B):
        remaining = B - step
        if step == 0:
            pool, _ = _lookahead_best_pool(current_p, u, G, n, remaining, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)

        if pool == 0:
            break
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool
        current_p = bayesian_update_single_test(current_p, pool, r, n)

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


# -------------------------------------------------------------------
# Counting-based greedy: full-history Bayesian update by counting
# -------------------------------------------------------------------

def greedy_myopic_counting_simulate(p, u, B, G, z_mask, pool_selector=None):
    """Simulate myopic greedy with full-history Bayesian update by counting.

    Like greedy_myopic_simulate but at each step computes posteriors by
    enumerating all 2^n infection profiles consistent with the FULL
    history, rather than applying sequential single-test updates.

    This uses bayesian_update_by_counting(p, history, n) which considers
    all historical data h_k jointly.

    Returns (history, cleared_mask, utility).
    """
    n = len(p)
    cleared_mask = 0
    history = ()

    for _ in range(B):
        # Compute posteriors from full history using counting
        if history:
            current_p = bayesian_update_by_counting(p, history, n)
        else:
            current_p = list(p)

        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            break
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


def greedy_myopic_gibbs_simulate(p, u, B, G, z_mask,
                                 num_iterations=1000, burn_in=200, seed=None,
                                 pool_selector=None):
    """Simulate myopic greedy with Gibbs sampling posterior updates.

    Like greedy_myopic_counting_simulate but uses Gibbs sampling (MCMC)
    to approximate posteriors instead of exact enumeration. Scales to
    larger populations (n~50+) where counting (O(2^n)) is infeasible.

    Returns (history, cleared_mask, utility).
    """
    n = len(p)
    cleared_mask = 0
    history = ()

    for _ in range(B):
        # Compute posteriors from full history using Gibbs sampling
        if history:
            current_p = gibbs_update(p, history, n,
                                     num_iterations=num_iterations,
                                     burn_in=burn_in, seed=seed)
        else:
            current_p = list(p)

        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            break
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


def greedy_myopic_gibbs_expected_utility(p, u, B, G,
                                          num_iterations=1000, burn_in=200,
                                          seed=42, pool_selector=None):
    """Expected utility of myopic greedy with Gibbs sampling posteriors.

    At each step picks the myopic-best pool using Gibbs-sampled posteriors,
    then recurses over all possible outcomes weighted by their probabilities.
    Uses a fixed seed for reproducibility within the recursive tree.
    """
    n = len(p)

    def recurse(prior_p, history, b, cleared_mask):
        if b == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        # Compute posteriors from full history via Gibbs
        if history:
            current_p = gibbs_update(prior_p, history, n,
                                     num_iterations=num_iterations,
                                     burn_in=burn_in, seed=seed)
        else:
            current_p = list(prior_p)

        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        pool_idx = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([current_p[i] for i in pool_idx])

        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_history = history + ((pool, r),)
            ev += pmf[r] * recurse(prior_p, new_history, b - 1, new_cleared)
        return ev

    return recurse(list(p), (), B, 0)


def greedy_myopic_counting_expected_utility(p, u, B, G, pool_selector=None):
    """Expected utility of myopic greedy with full-history Bayesian counting.

    At each step picks the myopic-best pool using posteriors computed by
    counting over all consistent worlds, then recurses over all possible
    outcomes weighted by their probabilities.
    """
    n = len(p)

    def recurse(prior_p, history, b, cleared_mask):
        if b == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        # Compute posteriors from full history
        if history:
            current_p = bayesian_update_by_counting(prior_p, history, n)
        else:
            current_p = list(prior_p)

        if pool_selector is not None:
            pool = pool_selector(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)
        if pool == 0:
            return sum(u[i] for i in indices_from_mask(cleared_mask, n))

        pool_idx = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([current_p[i] for i in pool_idx])

        ev = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_history = history + ((pool, r),)
            ev += pmf[r] * recurse(prior_p, new_history, b - 1, new_cleared)
        return ev

    return recurse(list(p), (), B, 0)
