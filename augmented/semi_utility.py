"""
Semi-utility meta-parameter for augmented adaptive group counting.

Extends the binary welfare model with a continuous blending parameter alpha
that interpolates between two extremes:

  - alpha=0: binary clearance model (only cleared individuals contribute utility)
  - alpha=1: utility proportional to posterior clearance probability
  - alpha in (0,1): convex combination of both

The semi-utility for a given state is:

  U_semi(alpha) = sum_i u_i * [alpha * P(clearancey_i | H_k) + (1-alpha) * 1_{cleared}(i)]

where P(clearancey_i | H_k) = 1 - p_posterior_i is the posterior probability that
individual i is clearancey given the test history H_k.

This file provides:
  1. semi_utility()                         — compute semi-utility for a given state
  2. greedy_myopic_semi_simulate()          — simulate greedy with semi-utility scoring
  3. greedy_myopic_semi_expected_utility()  — expected utility of the semi-utility greedy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import (all_pools, all_pools_from_mask, compute_active_mask,
                            indices_from_mask, test_result, mask_str)
from augmented.bayesian import (bayesian_update_single_test, bayesian_update,
                                bayesian_update_by_counting, gibbs_update,
                                _poisson_binomial_pmf)


# -------------------------------------------------------------------
# Semi-utility computation
# -------------------------------------------------------------------

def semi_utility(p_posterior, u, cleared_mask, n, alpha):
    """Compute the semi-utility for a given state.

    U_semi(alpha) = sum_i u_i * [alpha * P(clearancey_i | H_k) + (1-alpha) * 1_{cleared}(i)]

    Parameters
    ----------
    p_posterior : list[float]
        Posterior latent-state probabilities (length n).
        P(clearancey_i | H_k) = 1 - p_posterior[i].
    u : list[float]
        Utility weights for each individual (length n).
    cleared_mask : int
        Bitmask of cleared individuals (bit i set if individual i is cleared).
    n : int
        Population size.
    alpha : float
        Blending parameter in [0, 1].
        alpha=0 -> binary clearance only.
        alpha=1 -> posterior clearance probability only.

    Returns
    -------
    float
        The semi-utility value.
    """
    total = 0.0
    for i in range(n):
        p_clearancey = 1.0 - p_posterior[i]
        is_cleared = 1.0 if (cleared_mask >> i & 1) else 0.0
        total += u[i] * (alpha * p_clearancey + (1.0 - alpha) * is_cleared)
    return total


# -------------------------------------------------------------------
# Semi-utility pool scoring
# -------------------------------------------------------------------

def _semi_best_pool(p, u, G, n, cleared_mask, alpha, use_filtering=True):
    """Pick pool maximizing expected semi-utility after one test.

    Score(pool) = sum over r of P(r) * U_semi(posterior_after_r, alpha)

    where P(r) comes from the Poisson-Binomial PMF of the pool members'
    latent-state probabilities, and U_semi uses the updated posteriors.

    If use_filtering=True, only considers pools from active individuals
    (those not yet cleared and not confirmed active).
    """
    if use_filtering:
        active_mask, _ = compute_active_mask(p, cleared_mask, n)
        if active_mask == 0:
            return 0  # no uncertain individuals left
        pools = all_pools_from_mask(active_mask, G, include_empty=False)
    else:
        pools = all_pools(n, G, include_empty=False)

    best_pool, best_score = 0, -1.0

    for pool in pools:
        pool_idx = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([p[i] for i in pool_idx])

        score = 0.0
        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(p, pool, r, n)
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            score += pmf[r] * semi_utility(new_p, u, new_cleared, n, alpha)

        if score > best_score:
            best_score = score
            best_pool = pool

    return best_pool


# -------------------------------------------------------------------
# Greedy simulation with semi-utility scoring
# -------------------------------------------------------------------

def greedy_myopic_semi_simulate(p, u, B, G, z_mask, alpha,
                                update_method='sequential'):
    """Simulate greedy with semi-utility pool selection on a fixed latent-state profile.

    At each step: pick pool maximizing expected semi-utility (over all
    possible outcomes), observe augmented result, update beliefs, repeat.

    The pool selection uses semi_utility scoring with the alpha parameter.
    The final returned utility is the standard binary clearance utility
    (consistent with the rest of the codebase), so comparisons are fair.

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    u : list[float]
        Utility weights for each individual (length n).
    B : int
        Budget (number of tests).
    G : int
        Maximum pool size.
    z_mask : int
        True latent-state profile bitmask.
    alpha : float
        Semi-utility blending parameter in [0, 1].
    update_method : str
        Bayesian update method: 'sequential' (default), 'counting', or 'gibbs'.
        - 'sequential': apply single-test Bayesian updates sequentially.
        - 'counting': recompute posteriors from full history by counting.
        - 'gibbs': use Gibbs drawing to approximate posteriors from full history.

    Returns
    -------
    tuple
        (history, cleared_mask, utility) where utility is the standard
        binary clearance utility (sum of u_i for cleared individuals).
    """
    n = len(p)
    current_p = list(p)
    cleared_mask = 0
    history = ()

    for _ in range(B):
        # Compute posteriors based on update method
        if update_method == 'counting' and history:
            current_p = bayesian_update_by_counting(p, history, n)
        elif update_method == 'gibbs' and history:
            current_p = gibbs_update(p, history, n)
        # For 'sequential', current_p is already up-to-date from the loop

        pool = _semi_best_pool(current_p, u, G, n, cleared_mask, alpha)
        if pool == 0:
            break  # no useful pool
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool

        # Update posteriors for sequential method
        if update_method == 'sequential':
            current_p = bayesian_update_single_test(current_p, pool, r, n)

    utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, utility


# -------------------------------------------------------------------
# Expected utility of the semi-utility greedy
# -------------------------------------------------------------------

def greedy_myopic_semi_expected_utility(p, u, B, G, alpha,
                                        update_method='sequential'):
    """Expected utility of the semi-utility greedy strategy.

    At each step picks the pool maximizing expected semi-utility (with
    alpha blending), then recurses over all possible outcomes weighted
    by their probabilities.

    The returned value is the expected BINARY clearance utility (standard
    metric), not the semi-utility.  The semi-utility is only used for
    pool selection.

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    u : list[float]
        Utility weights for each individual (length n).
    B : int
        Budget (number of tests).
    G : int
        Maximum pool size.
    alpha : float
        Semi-utility blending parameter in [0, 1].
    update_method : str
        Bayesian update method: 'sequential', 'counting', or 'gibbs'.

    Returns
    -------
    float
        Expected binary clearance utility under the semi-utility greedy.
    """
    n = len(p)

    if update_method == 'sequential':
        def recurse(current_p, b, cleared_mask):
            if b == 0:
                return sum(u[i] for i in indices_from_mask(cleared_mask, n))

            pool = _semi_best_pool(current_p, u, G, n, cleared_mask, alpha)
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

    else:
        # For 'counting' and 'gibbs', posteriors depend on full history
        def recurse(prior_p, history, b, cleared_mask):
            if b == 0:
                return sum(u[i] for i in indices_from_mask(cleared_mask, n))

            # Compute posteriors from full history
            if history:
                if update_method == 'counting':
                    current_p = bayesian_update_by_counting(prior_p, history, n)
                else:  # gibbs
                    current_p = gibbs_update(prior_p, history, n)
            else:
                current_p = list(prior_p)

            pool = _semi_best_pool(current_p, u, G, n, cleared_mask, alpha)
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
