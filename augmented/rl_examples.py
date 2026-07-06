"""
Augmented DAPTS as a finite-horizon Markov Decision Process (MDP),
with two pedagogical "first RL examples":

  1. value_iteration(p, u, B, G):
     Explicit Bellman value iteration (backward sweep) over enumerated
     information states. Mathematically identical to the existing DP
     solver in augmented/solver.py, but written in MDP vocabulary:
     states, actions, transitions, rewards, Bellman equation. Intended
     as the first bridge from "dynamic programming over trees" to
     "value iteration on an MDP".

  2. tabular_q_learning(p, u, B, G, ...):
     Classic Q-learning with ε-greedy exploration. Draws episodes
     (one ground-truth latent-state profile per episode), interacts with
     the DAPTS environment, and updates Q(state, action) via TD(0).
     Converges to the Q* produced by value_iteration.

MDP formulation
---------------
State s = (k, remaining, cleared_mask)
  * k:              number of tests already used (0..B)
  * remaining:      frozenset of latent-state profiles z consistent with
                    the observed history
  * cleared_mask:   bitmask of individuals proven clearancey so far
Action a = pool mask (subset of [n] of size <= G; a = 0 means "wait")
Transition: choosing pool a when state is (k, remaining, cleared)
  * Observe r = |a ∩ Z|, with Z ~ prior restricted to `remaining`.
  * Next remaining = { z ∈ remaining : test_result(a, z) = r }.
  * Next cleared = cleared ∪ a if r = 0 else cleared.
Reward: 0 at all non-terminal steps; at step B, reward = sum_{i in cleared} u_i.
Objective: maximize expected terminal reward.

Under this formulation, V*(initial_state) equals the optimal DAPTS
expected utility, which matches solve_optimal_dapts exactly.
"""

import random

from augmented.core import all_pools, test_result


# -------------------------------------------------------------------
# MDP helpers
# -------------------------------------------------------------------

def _prior_weights(p, n):
    """Return w[z] = Pr(Z = z) under the independent prior for all z."""
    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n
    w = [0.0] * num_profiles
    for z in range(num_profiles):
        wz = 1.0
        for i in range(n):
            wz *= p[i] if (z >> i & 1) else q[i]
        w[z] = wz
    return w


def _cleared_utility(cleared_mask, u, n):
    """Total utility of individuals marked cleared."""
    return sum(u[i] for i in range(n) if (cleared_mask >> i & 1))


def _transition(remaining, cleared, pool, w):
    """Partition a transition by observed outcome r.

    Returns a list of (r, mass_r, new_remaining, new_cleared) entries
    with mass_r = sum of prior weights of profiles in `remaining` that
    would produce outcome r when pool is tested.
    """
    buckets = {}
    for z in remaining:
        r = test_result(pool, z)
        buckets.setdefault(r, []).append(z)

    out = []
    for r, zs in buckets.items():
        mass_r = sum(w[z] for z in zs)
        new_cleared = cleared | pool if r == 0 else cleared
        out.append((r, mass_r, frozenset(zs), new_cleared))
    return out


# -------------------------------------------------------------------
# Example 1: Value iteration on the information-state MDP
# -------------------------------------------------------------------

def value_iteration(p, u, B, G):
    """Bellman value iteration (= backward induction) over enumerated
    information states.

    Because the horizon is finite and every state belongs to exactly
    one step k, the MDP admits a single backward sweep, so "value
    iteration" here is really finite-horizon backward induction.

    Returns
    -------
    V : dict
        (k, remaining, cleared) -> V*(state), the optimal value.
    Q : dict
        (k, remaining, cleared, pool) -> Q*(state, action).
    pi : dict
        (k, remaining, cleared) -> argmax pool.

    Notes
    -----
    Values are CONDITIONAL: V(s) is the expected future utility given
    the agent is at s. In particular V*(initial_state) is the optimal
    DAPTS value, matching solve_optimal_dapts to machine precision.
    """
    n = len(p)
    w = _prior_weights(p, n)
    pools = all_pools(n, G, include_empty=True)
    all_z = frozenset(range(1 << n))

    # ----- Forward reach: enumerate states reachable at each step -----
    states_at = [set() for _ in range(B + 1)]
    states_at[0].add((all_z, 0))
    for k in range(B):
        for remaining, cleared in states_at[k]:
            for a in pools:
                for _r, _m, new_remaining, new_cleared in \
                        _transition(remaining, cleared, a, w):
                    states_at[k + 1].add((new_remaining, new_cleared))

    # ----- Backward Bellman sweep -----
    V, Q, pi = {}, {}, {}

    # Terminal layer: V = utility of individuals cleared so far
    for remaining, cleared in states_at[B]:
        V[(B, remaining, cleared)] = _cleared_utility(cleared, u, n)

    # Bellman backup for k = B-1, B-2, ..., 0
    for k in range(B - 1, -1, -1):
        for remaining, cleared in states_at[k]:
            total_mass = sum(w[z] for z in remaining)
            best_val = -float('inf')
            best_a = 0

            for a in pools:
                ev = 0.0
                for _r, mass_r, new_rem, new_cl in \
                        _transition(remaining, cleared, a, w):
                    prob_r = mass_r / total_mass if total_mass > 0 else 0.0
                    ev += prob_r * V[(k + 1, new_rem, new_cl)]

                Q[(k, remaining, cleared, a)] = ev
                if ev > best_val:
                    best_val = ev
                    best_a = a

            V[(k, remaining, cleared)] = best_val
            pi[(k, remaining, cleared)] = best_a

    return V, Q, pi


def value_iteration_optimal_value(p, u, B, G):
    """Convenience wrapper: V*(initial) as a single float."""
    V, _, _ = value_iteration(p, u, B, G)
    n = len(p)
    initial = (0, frozenset(range(1 << n)), 0)
    return V[initial]


# -------------------------------------------------------------------
# Example 2: Tabular Q-learning (model-free) on the same MDP
# -------------------------------------------------------------------

def _draw_profile(p, rng):
    z = 0
    for i, pi in enumerate(p):
        if rng.random() < pi:
            z |= 1 << i
    return z


def tabular_q_learning(p, u, B, G, num_episodes=5000, alpha='auto',
                       epsilon=0.3, seed=0):
    """Tabular Q-learning with ε-greedy exploration.

    No knowledge of the transition/reward model is assumed: the agent
    only interacts with the DAPTS environment episode by episode. After
    enough episodes, Q(state, action) approaches the Q* returned by
    value_iteration.

    Parameters
    ----------
    num_episodes : int
        Number of training episodes (each draws one Z and plays B tests).
    alpha : float or 'auto'
        Step size. If 'auto', uses the Robbins-Monro schedule
        α = 1 / (1 + N(s,a)), where N(s,a) is the visit count INCLUDING
        the current visit (incremented before the step-size is read).
        This schedule guarantees convergence of tabular Q-learning.
        Fixed α (e.g. 0.1) runs the usual "slow average" variant but
        does not converge to Q* exactly.
    epsilon : float
        Exploration probability for ε-greedy action selection.
    seed : int
        RNG seed.

    Returns
    -------
    Q : dict
        (state, action) -> estimated Q value. Only (state, action) pairs
        actually visited during training are present.
    """
    n = len(p)
    rng = random.Random(seed)
    pools = all_pools(n, G, include_empty=True)
    all_z = frozenset(range(1 << n))

    Q = {}
    N = {}  # visit counts, used only when alpha == 'auto'

    def q_get(s, a):
        return Q.get((s, a), 0.0)

    def greedy_action(s):
        best_a = pools[0]
        best_v = q_get(s, best_a)
        for a in pools[1:]:
            v = q_get(s, a)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a

    def step_size(s, a):
        if alpha == 'auto':
            N[(s, a)] = N.get((s, a), 0) + 1
            return 1.0 / (1.0 + N[(s, a)])
        return alpha

    for _ in range(num_episodes):
        z_true = _draw_profile(p, rng)
        remaining = all_z
        cleared = 0

        for k in range(B):
            s = (k, remaining, cleared)

            # ε-greedy
            if rng.random() < epsilon:
                a = rng.choice(pools)
            else:
                a = greedy_action(s)

            # Environment step (the true z is fixed for this episode)
            r = test_result(a, z_true)
            new_remaining = frozenset(zz for zz in remaining
                                      if test_result(a, zz) == r)
            new_cleared = cleared | a if r == 0 else cleared
            s_next = (k + 1, new_remaining, new_cleared)

            # TD target: at terminal, use the utility of cleared_mask;
            # otherwise, bootstrap with max_a' Q(s', a').
            if k == B - 1:
                target = _cleared_utility(new_cleared, u, n)
            else:
                target = max(q_get(s_next, aa) for aa in pools)

            # Q-learning update
            a_step = step_size(s, a)
            Q[(s, a)] = q_get(s, a) + a_step * (target - q_get(s, a))

            remaining, cleared = new_remaining, new_cleared

    return Q


def q_learning_policy_value(p, u, B, G, Q):
    """Evaluate the greedy policy extracted from Q by full enumeration.

    Runs the Bellman equation once using the policy a(s) = argmax_a Q(s,a)
    (with deterministic tie-breaking). Returns the expected utility at
    the initial state — comparable to value_iteration_optimal_value.
    """
    n = len(p)
    w = _prior_weights(p, n)
    pools = all_pools(n, G, include_empty=True)
    all_z = frozenset(range(1 << n))

    def policy_at(s):
        best_a, best_v = pools[0], Q.get((s, pools[0]), 0.0)
        for a in pools[1:]:
            v = Q.get((s, a), 0.0)
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    memo = {}

    def value(k, remaining, cleared):
        key = (k, remaining, cleared)
        if key in memo:
            return memo[key]

        if k == B:
            memo[key] = _cleared_utility(cleared, u, n)
            return memo[key]

        a = policy_at(key)
        total_mass = sum(w[z] for z in remaining)
        ev = 0.0
        for _r, mass_r, new_rem, new_cl in \
                _transition(remaining, cleared, a, w):
            prob_r = mass_r / total_mass if total_mass > 0 else 0.0
            ev += prob_r * value(k + 1, new_rem, new_cl)

        memo[key] = ev
        return ev

    return value(0, all_z, 0)
