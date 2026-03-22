"""
Hybrid greedy -> brute-force solver for augmented pooled testing.

Combines greedy pool selection for the first K steps with exact DP
(solve_optimal_dapts) for the remaining B-K steps.  Includes
entropy-based scoring and branch value estimation.

Part A: Entropy, information gain, infection-aware scoring
Part B: Branch value estimation (lower/upper bounds)
Part C: Core hybrid solver
"""

import math

from augmented.core import (
    mask_from_indices, indices_from_mask, popcount, mask_str,
    all_pools_from_mask, compute_active_mask,
)
from augmented.bayesian import (
    bayesian_update_single_test, _poisson_binomial_pmf,
)
from augmented.greedy import _myopic_best_pool, greedy_myopic_expected_utility
from augmented.solver import solve_optimal_dapts
from augmented.tree_extractor import extract_tree
from augmented.baselines import u_max


# ===================================================================
# Part A: Entropy, information gain, infection-aware scoring
# ===================================================================

def _safe_binary_entropy(x):
    """Binary entropy H(x) in bits.  0*log(0) = 0 convention."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log2(x) + (1.0 - x) * math.log2(1.0 - x))


def expected_info_gain(pool_mask, p, n):
    """Expected reduction in posterior entropy from testing pool.

    H_before - E[H_after | outcomes].

    Uses _poisson_binomial_pmf and bayesian_update_single_test.
    """
    pool_indices = indices_from_mask(pool_mask, n)
    if not pool_indices:
        return 0.0

    # Current entropy (sum of per-agent binary entropies)
    h_before = sum(_safe_binary_entropy(p[i]) for i in range(n))

    # PMF over outcomes r = 0, ..., |pool|
    pool_probs = [p[i] for i in pool_indices]
    pmf = _poisson_binomial_pmf(pool_probs)

    # Expected posterior entropy
    e_h_after = 0.0
    for r in range(len(pool_indices) + 1):
        pr = pmf[r]
        if pr < 1e-15:
            continue
        p_post = bayesian_update_single_test(p, pool_mask, r, n)
        h_after = sum(_safe_binary_entropy(p_post[i]) for i in range(n))
        e_h_after += pr * h_after

    return h_before - e_h_after


def infection_aware_score(pool_mask, p, u, n, cleared_mask, alpha=0.5):
    """Score = alpha * P(r=0)*sum(u_i uncleared) + (1-alpha) * expected_info_gain.

    Blends the standard myopic gain with information-theoretic value.
    """
    pool_indices = indices_from_mask(pool_mask, n)
    if not pool_indices:
        return 0.0

    # Myopic component: P(r=0) * sum of utility for uncleared pool members
    prob_clear = 1.0
    for i in pool_indices:
        prob_clear *= (1.0 - p[i])
    gain = sum(u[i] for i in pool_indices if not (cleared_mask >> i & 1))
    myopic = prob_clear * gain

    # Information gain component
    ig = expected_info_gain(pool_mask, p, n)

    return alpha * myopic + (1.0 - alpha) * ig


# ===================================================================
# Part B: Branch value estimation
# ===================================================================

def estimate_branch_value(p_posterior, u, remaining_B, G, cleared_mask, n):
    """Return (lower_bound, upper_bound) for expected utility from this state.

    lower = greedy_myopic_expected_utility on the subproblem
    upper = cleared_utility + u_max(active_p, active_u)

    Uses compute_active_mask to identify active agents.
    Numerical safety: if lower > upper + 1e-9, fall back to safe bounds.
    """
    # Utility already secured from cleared agents
    cleared_indices = indices_from_mask(cleared_mask, n)
    cleared_utility = sum(u[i] for i in cleared_indices)

    # Identify active agents (uncertain infection status)
    active_mask, confirmed_infected_mask = compute_active_mask(
        p_posterior, cleared_mask, n
    )
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        # No uncertain agents remain
        return cleared_utility, cleared_utility

    # Build active subproblem
    active_p = [p_posterior[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    # Upper bound: cleared utility + u_max on active subproblem
    upper = cleared_utility + u_max(active_p, active_u)

    # Lower bound: cleared utility + greedy on active subproblem
    sub_G = min(G, n_active)
    lower = cleared_utility + greedy_myopic_expected_utility(
        active_p, active_u, remaining_B, sub_G
    )

    # Numerical safety
    if lower > upper + 1e-9:
        safe_lb = cleared_utility
        safe_ub = cleared_utility + sum(active_u)
        return safe_lb, safe_ub

    return lower, upper


# ===================================================================
# Part C: Hybrid solver helpers
# ===================================================================

def _infection_aware_best_pool(p, u, G, n, cleared_mask, alpha):
    """Pick pool maximizing infection_aware_score."""
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    if active_mask == 0:
        return 0
    pools = all_pools_from_mask(active_mask, G, include_empty=False)
    if not pools:
        return 0

    best_pool, best_score = 0, -1.0
    for pool in pools:
        s = infection_aware_score(pool, p, u, n, cleared_mask, alpha)
        if s > best_score:
            best_score = s
            best_pool = pool
    return best_pool


def _adjust_steps(tree, offset):
    """Add offset to all step numbers (in-place)."""
    if 'step' in tree:
        tree['step'] += offset
    if 'children' in tree:
        for child in tree['children'].values():
            _adjust_steps(child, offset)


def _update_history(tree, prefix_history):
    """Prepend prefix_history to all history tuples (in-place)."""
    if 'history' in tree:
        tree['history'] = prefix_history + tree['history']
    if 'children' in tree:
        for child in tree['children'].values():
            _update_history(child, prefix_history)


def _remap_tree_indices(tree, idx_map, n, full_p, cleared_mask, u):
    """Remap sub-problem indices back to original population (in-place).

    idx_map: dict sub_index -> original_index.
    Must remap: pool masks, cleared masks, posteriors, pool_str, cleared_str,
    history tuples, terminal utilities.
    """
    sub_n = len(idx_map)

    # Remap posteriors: build full-population posterior
    if 'posteriors' in tree:
        sub_posteriors = tree['posteriors']
        full_posteriors = list(full_p)  # start from full prior/current
        # Cleared agents have p=0
        for i in indices_from_mask(cleared_mask, n):
            full_posteriors[i] = 0.0
        # Map sub-posteriors to original indices
        for sub_i, orig_i in idx_map.items():
            if sub_i < len(sub_posteriors):
                full_posteriors[orig_i] = sub_posteriors[sub_i]
        tree['posteriors'] = full_posteriors

    # Remap pool mask
    if 'pool' in tree:
        sub_pool = tree['pool']
        new_pool = 0
        for sub_i in indices_from_mask(sub_pool, sub_n):
            if sub_i in idx_map:
                new_pool |= 1 << idx_map[sub_i]
        tree['pool'] = new_pool
        tree['pool_str'] = mask_str(new_pool, n)

    # Remap cleared mask
    if 'cleared' in tree:
        sub_cleared = tree['cleared']
        # Start with the already-cleared mask from the parent phase
        new_cleared = cleared_mask
        for sub_i in indices_from_mask(sub_cleared, sub_n):
            if sub_i in idx_map:
                new_cleared |= 1 << idx_map[sub_i]
        tree['cleared'] = new_cleared
        tree['cleared_str'] = mask_str(new_cleared, n)

    # Remap history tuples
    if 'history' in tree:
        new_history = []
        for pool_mask_h, r_h in tree['history']:
            new_pool_h = 0
            for sub_i in indices_from_mask(pool_mask_h, sub_n):
                if sub_i in idx_map:
                    new_pool_h |= 1 << idx_map[sub_i]
            new_history.append((new_pool_h, r_h))
        tree['history'] = tuple(new_history)

    # Recompute terminal utility based on full cleared mask
    if tree.get('terminal') and 'utility' in tree:
        tree['utility'] = sum(u[i] for i in indices_from_mask(tree['cleared'], n))

    # Recurse into children
    if 'children' in tree:
        for child in tree['children'].values():
            _remap_tree_indices(child, idx_map, n, full_p, cleared_mask, u)


# ===================================================================
# Part C: Core hybrid solver
# ===================================================================

def hybrid_greedy_bruteforce(p, u, B, G, greedy_steps,
                              greedy_score_fn=None,
                              update_method='sequential',
                              pool_selector=None):
    """Hybrid solver: greedy for first K steps, then exact DP.

    Returns (tree_dict, expected_utility).
    tree_dict matches tree_extractor.extract_tree schema.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities.
    u : list[float]
        Individual utilities.
    B : int
        Total budget (number of tests).
    G : int
        Maximum pool size.
    greedy_steps : int
        Number of greedy steps (K) before switching to DP.
    greedy_score_fn : callable or None
        If provided, called as greedy_score_fn(p, u, G, n, cleared_mask)
        to select the pool at each greedy step.  Returns pool mask.
    update_method : str
        'sequential' for standard Bayesian updates.
    pool_selector : callable or None
        If provided and greedy_score_fn is None, used as the pool
        selection function.  Same signature as greedy_score_fn.
    """
    n = len(p)
    K = min(greedy_steps, B)

    # If pool_selector provided, wrap it as greedy_score_fn
    if pool_selector is not None and greedy_score_fn is None:
        greedy_score_fn = pool_selector

    # Special case: full DP (K=0)
    if K == 0:
        if n > 14:
            # Fall back to full greedy
            return _full_greedy_tree(p, u, B, G, n, greedy_score_fn)
        val, policy = solve_optimal_dapts(p, u, B, G)
        tree = extract_tree(policy, p, u, n)
        return tree, val

    # Special case: full greedy (K=B)
    if K >= B:
        tree, eu = _full_greedy_tree(p, u, B, G, n, greedy_score_fn)
        return tree, eu

    # Hybrid: greedy for K steps, then DP for remaining B-K
    tree, eu = _hybrid_recurse(
        p, u, B, G, n, K, greedy_score_fn,
        current_p=list(p),
        cleared_mask=0,
        step=1,
        history=(),
        remaining_greedy=K,
        remaining_budget=B,
    )
    return tree, eu


def _full_greedy_tree(p, u, B, G, n, greedy_score_fn):
    """Build full greedy decision tree and compute expected utility.

    Returns (tree_dict, expected_utility).
    """
    def _recurse(current_p, cleared_mask, step, history, remaining_b):
        if remaining_b == 0:
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': step,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }, utility

        # Choose pool
        if greedy_score_fn is not None:
            pool = greedy_score_fn(current_p, u, G, n, cleared_mask)
        else:
            pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)

        if pool == 0:
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': step,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }, utility

        pool_indices = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([current_p[i] for i in pool_indices])

        children = {}
        ev = 0.0
        for r in range(len(pool_indices) + 1):
            pr = pmf[r]
            if pr < 1e-15:
                continue
            new_p = bayesian_update_single_test(current_p, pool, r, n)
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_history = history + ((pool, r),)
            child, child_eu = _recurse(
                new_p, new_cleared, step + 1, new_history, remaining_b - 1
            )
            children[r] = child
            ev += pr * child_eu

        tree = {
            'step': step,
            'terminal': False,
            'pool': pool,
            'pool_str': mask_str(pool, n),
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'children': children,
        }
        return tree, ev

    return _recurse(list(p), 0, 1, (), B)


def _hybrid_recurse(p, u, B, G, n, K, greedy_score_fn,
                     current_p, cleared_mask, step, history,
                     remaining_greedy, remaining_budget):
    """Recursive hybrid: greedy phase then DP phase.

    Returns (tree_node, expected_utility).
    """
    # Phase 2: Switch to DP
    if remaining_greedy == 0:
        return _dp_phase(
            p, u, G, n, current_p, cleared_mask, step,
            history, remaining_budget, greedy_score_fn
        )

    # Phase 1: Greedy step
    if remaining_budget == 0:
        utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return {
            'step': step,
            'terminal': True,
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'utility': utility,
        }, utility

    # Choose pool greedily
    if greedy_score_fn is not None:
        pool = greedy_score_fn(current_p, u, G, n, cleared_mask)
    else:
        pool = _myopic_best_pool(current_p, u, G, n, cleared_mask)

    if pool == 0:
        # No useful pool; go straight to DP or terminal
        if remaining_budget > 0 and remaining_greedy > 0:
            return _dp_phase(
                p, u, G, n, current_p, cleared_mask, step,
                history, remaining_budget, greedy_score_fn
            )
        utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return {
            'step': step,
            'terminal': True,
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'utility': utility,
        }, utility

    pool_indices = indices_from_mask(pool, n)
    pmf = _poisson_binomial_pmf([current_p[i] for i in pool_indices])

    children = {}
    ev = 0.0
    for r in range(len(pool_indices) + 1):
        pr = pmf[r]
        if pr < 1e-15:
            continue
        new_p = bayesian_update_single_test(current_p, pool, r, n)
        new_cleared = cleared_mask | pool if r == 0 else cleared_mask
        new_history = history + ((pool, r),)

        child, child_eu = _hybrid_recurse(
            p, u, B, G, n, K, greedy_score_fn,
            current_p=new_p,
            cleared_mask=new_cleared,
            step=step + 1,
            history=new_history,
            remaining_greedy=remaining_greedy - 1,
            remaining_budget=remaining_budget - 1,
        )
        children[r] = child
        ev += pr * child_eu

    tree = {
        'step': step,
        'terminal': False,
        'pool': pool,
        'pool_str': mask_str(pool, n),
        'history': history,
        'cleared': cleared_mask,
        'cleared_str': mask_str(cleared_mask, n),
        'posteriors': list(current_p),
        'children': children,
    }
    return tree, ev


def _dp_phase(p, u, G, n, current_p, cleared_mask, step,
              history, remaining_budget, greedy_score_fn):
    """Switch to exact DP on a reduced subproblem of active agents.

    Steps:
    1. Use compute_active_mask to get active agents
    2. Build sub_p, sub_u with only active agents
    3. Call solve_optimal_dapts(sub_p, sub_u, remaining_B, min(G, sub_n))
    4. extract_tree on the sub-policy
    5. Remap sub-tree indices back to original population
    6. Add cleared_utility to DP value

    If n_active > 14, fall back to continued greedy.
    """
    if remaining_budget == 0:
        utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return {
            'step': step,
            'terminal': True,
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'utility': utility,
        }, utility

    # Compute cleared utility
    cleared_indices = indices_from_mask(cleared_mask, n)
    cleared_utility = sum(u[i] for i in cleared_indices)

    # Identify active agents
    active_mask, confirmed_infected_mask = compute_active_mask(
        current_p, cleared_mask, n
    )
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        # No uncertain agents, return terminal
        utility = cleared_utility
        return {
            'step': step,
            'terminal': True,
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'utility': utility,
        }, utility

    # If too many active agents for DP, fall back to continued greedy
    if n_active > 14:
        return _greedy_fallback(
            u, G, n, current_p, cleared_mask, step,
            history, remaining_budget, greedy_score_fn
        )

    # Build reduced subproblem
    sub_p = [current_p[i] for i in active_indices]
    sub_u = [u[i] for i in active_indices]
    sub_n = n_active
    sub_G = min(G, sub_n)

    # Solve exact DP on reduced subproblem
    dp_val, dp_policy = solve_optimal_dapts(sub_p, sub_u, remaining_budget, sub_G)

    # Extract tree from sub-policy
    sub_tree = extract_tree(dp_policy, sub_p, sub_u, sub_n)

    # Build index mapping: sub_index -> original_index
    idx_map = {sub_i: orig_i for sub_i, orig_i in enumerate(active_indices)}

    # Remap the sub-tree indices to original population
    _remap_tree_indices(sub_tree, idx_map, n, current_p, cleared_mask, u)

    # Adjust step numbers: sub-tree starts at step 1, we need step
    _adjust_steps(sub_tree, step - 1)

    # Prepend history from greedy phase
    _update_history(sub_tree, history)

    # Total expected utility = cleared utility + DP value
    total_eu = cleared_utility + dp_val

    return sub_tree, total_eu


def _greedy_fallback(u, G, n, current_p, cleared_mask, step,
                      history, remaining_budget, greedy_score_fn):
    """Fall back to greedy when DP is infeasible (n_active > 14)."""
    def _recurse(cp, cm, s, h, rb):
        if rb == 0:
            utility = sum(u[i] for i in indices_from_mask(cm, n))
            return {
                'step': s,
                'terminal': True,
                'history': h,
                'cleared': cm,
                'cleared_str': mask_str(cm, n),
                'posteriors': list(cp),
                'utility': utility,
            }, utility

        if greedy_score_fn is not None:
            pool = greedy_score_fn(cp, u, G, n, cm)
        else:
            pool = _myopic_best_pool(cp, u, G, n, cm)

        if pool == 0:
            utility = sum(u[i] for i in indices_from_mask(cm, n))
            return {
                'step': s,
                'terminal': True,
                'history': h,
                'cleared': cm,
                'cleared_str': mask_str(cm, n),
                'posteriors': list(cp),
                'utility': utility,
            }, utility

        pool_indices = indices_from_mask(pool, n)
        pmf = _poisson_binomial_pmf([cp[i] for i in pool_indices])

        children = {}
        ev = 0.0
        for r in range(len(pool_indices) + 1):
            pr = pmf[r]
            if pr < 1e-15:
                continue
            new_p = bayesian_update_single_test(cp, pool, r, n)
            new_cleared = cm | pool if r == 0 else cm
            new_history = h + ((pool, r),)
            child, child_eu = _recurse(new_p, new_cleared, s + 1,
                                        new_history, rb - 1)
            children[r] = child
            ev += pr * child_eu

        tree = {
            'step': s,
            'terminal': False,
            'pool': pool,
            'pool_str': mask_str(pool, n),
            'history': h,
            'cleared': cm,
            'cleared_str': mask_str(cm, n),
            'posteriors': list(cp),
            'children': children,
        }
        return tree, ev

    return _recurse(current_p, cleared_mask, step, history, remaining_budget)
