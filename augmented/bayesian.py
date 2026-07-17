"""
Bayesian posterior update for augmented grouped tests.

Given prior latent-state probabilities p = (p_1, ..., p_n) and a test history
H_k = ((t_1, r_1), ..., (t_k, r_k)), compute posterior probabilities
q'_i = P(individual i is clearancey | H_k) for each individual.

In the idealized augmented model, counting pool t yields r = |t ∩ Z|
(the exact count of active in the pool).
"""

import math
import warnings

from augmented.core import indices_from_mask, test_result, popcount


def _poisson_binomial_pmf(probs):
    """PMF of the Poisson-Binomial distribution for independent Bernoullis.

    Given probabilities [p_1, ..., p_m], returns a list pmf of length m+1
    where pmf[k] = P(exactly k successes).

    Uses the DP recurrence:
        dp[0] = 1
        dp[k] = dp[k] * (1-p_j) + dp[k-1] * p_j   (for each new p_j)
    """
    m = len(probs)
    dp = [0.0] * (m + 1)
    dp[0] = 1.0
    for j, pj in enumerate(probs):
        # Traverse backwards to avoid overwriting values we still need
        for k in range(j + 1, 0, -1):
            dp[k] = dp[k] * (1.0 - pj) + dp[k - 1] * pj
        dp[0] *= (1.0 - pj)
    return dp


def bayesian_update_single_test(p, pool_mask, r, n):
    """Update latent-state probabilities after one augmented test.

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    pool_mask : int
        Bitmask of the tested pool t.
    r : int
        Observed result r = |t ∩ Z| (count of active in pool).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior latent-state probabilities p'_i = P(Z_i=1 | r, t).

    Math
    ----
    For i NOT in t: test gives no info, so p'_i = p_i.

    For i IN t, by Bayes:
        p'_i = P(Z_i=1 | r) = P(r | Z_i=1) * p_i / P(r)

    where (letting S = t \\ {i}):
        P(r | Z_i=1) = P(exactly r-1 active in S)   [Poisson-Binomial]
        P(r | Z_i=0) = P(exactly r   active in S)   [Poisson-Binomial]
        P(r)          = P(r|Z_i=1)*p_i + P(r|Z_i=0)*q_i
    """
    pool_indices = indices_from_mask(pool_mask, n)

    if not pool_indices:
        return list(p)

    posterior = list(p)

    # For each i in pool, compute posterior via Poisson-Binomial on t\{i}
    for i in pool_indices:
        # Handle deterministic cases: p_i = 0 or p_i = 1
        if p[i] <= 0.0:
            posterior[i] = 0.0
            continue
        if p[i] >= 1.0:
            posterior[i] = 1.0
            continue

        others = [j for j in pool_indices if j != i]
        others_p = [max(0.0, min(1.0, p[j])) for j in others]

        # PMF of number of active among others
        pmf = _poisson_binomial_pmf(others_p)

        # P(r | Z_i = 1) = P(r-1 active among others)
        p_r_given_1 = pmf[r - 1] if r >= 1 else 0.0
        # P(r | Z_i = 0) = P(r active among others)
        p_r_given_0 = pmf[r] if r <= len(others) else 0.0

        # Bayes
        numerator = p_r_given_1 * p[i]
        denominator = numerator + p_r_given_0 * (1.0 - p[i])

        if denominator > 0:
            posterior[i] = numerator / denominator
        # else: degenerate case, keep prior

    return posterior


def bayesian_update(p, history, n):
    """Apply Bayesian updates for a full test history (sequential).

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior latent-state probabilities after all tests in history.
    """
    current_p = list(p)
    for pool_mask, r in history:
        current_p = bayesian_update_single_test(current_p, pool_mask, r, n)
    return current_p


def bayesian_update_by_counting(p, history, n):
    """Compute posterior P(Z_i=1 | h_k) by counting over all consistent worlds.

    This is the "by counting" approach: enumerate all 2^n latent-state profiles,
    keep those consistent with the full test history, and compute posteriors
    as weighted proportions.

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Full test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior latent-state probabilities P(Z_i=1 | h_k).

    Notes
    -----
    Complexity: O(2^n * k) where k = len(history).
    Feasible for n <= ~20. This is the EXACT joint posterior. It coincides
    with sequential single-test updates (`bayesian_update`) only when the
    tested pools are pairwise disjoint; with OVERLAPPING pools the two differ,
    because the sequential update treats marginals as independent and so misses
    the cross-test deductions that counting captures (e.g. tests {0,1}=1 and
    {1,2}=0 force individual 0 active — counting sees it, sequential does not).
    """
    if not history:
        return list(p)

    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n

    # Accumulate: weighted count of consistent profiles, and per-individual
    # weighted count of consistent profiles where Z_i = 1
    total_weight = 0.0
    active_weight = [0.0] * n

    for z_mask in range(num_profiles):
        # Check consistency with ALL tests in history
        consistent = True
        for pool_mask, r in history:
            if test_result(pool_mask, z_mask) != r:
                consistent = False
                break

        if not consistent:
            continue

        # Pr(Z = z_mask) under the independent prior
        w = 1.0
        for i in range(n):
            w *= p[i] if (z_mask >> i & 1) else q[i]

        total_weight += w
        # Add weight to each active individual in this profile
        bits = z_mask
        while bits:
            lsb = bits & -bits
            i = lsb.bit_length() - 1
            active_weight[i] += w
            bits ^= lsb

    # Compute posteriors. total_weight==0 means NO latent-state profile is
    # consistent with the history (or all consistent profiles have zero prior
    # probability). Returning the prior here would silently hide a bug, so raise.
    if total_weight <= 0.0:
        raise ValueError(
            "bayesian_update_by_counting: infeasible history — no latent_state "
            "profile is consistent with it (or all have zero prior mass). "
            "Refusing to silently return the prior."
        )
    posterior = list(p)
    for i in range(n):
        posterior[i] = active_weight[i] / total_weight

    return posterior


def exact_pool_pmf(p, history, pool_mask, n):
    """Exact posterior PMF of r_t = |t ∩ Z| given history H.

    Enumerates all 2^n profiles, restricts to those consistent with H, and
    aggregates prior weights by the outcome r_t they produce. Returns a list of
    length popcount(pool_mask)+1 where entry k is P(r_t = k | H). If no profile
    is consistent with the history, the returned PMF has total mass 0.

    This is the CORRECT branch-weight distribution for evaluating the expected
    utility of a counting/gibbs greedy policy: after conditioning on H the joint
    posterior over (Z_i)_{i in t} is correlated, so the Poisson-Binomial of the
    posterior MARGINALS is NOT the true distribution of r_t (it can put nonzero_count
    mass on impossible counts and mis-weight the recursion).
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


# Per-CONNECTED-COMPONENT exact enumeration cap. A component with this many or
# fewer agents is solved exactly (cost 2^|component|, independent of n); larger
# single components fall back to the alternating-move MCMC. Because the cap is
# per component (not over the whole active set) it covers every real scale of the
# project (the exact DP itself tops out at n<=14).
EXACT_ACTIVE_THRESHOLD = 16


def _find_valid_state(remaining_tests, active_list, p, rng,
                      restarts=200, steps=1000):
    """Busca una asignacion sobre el conjunto activo consistente con TODOS los
    ``remaining_tests`` (conteos exactos), por busqueda local de minimos
    conflictos con reinicios. Devuelve {agente: 0/1} o None si no halla una en el
    presupuesto (la historia se asume factible, asi que normalmente exito rapido).
    """
    A = active_list
    test_members = [[j for j in A if pm >> j & 1] for pm, _ in remaining_tests]
    targets = [r for _, r in remaining_tests]

    def total_violation(state):
        return sum(abs(sum(state[j] for j in members) - r)
                   for members, r in zip(test_members, targets))

    for _ in range(restarts):
        state = {i: (1 if rng.random() < p[i] else 0) for i in A}
        v = total_violation(state)
        for _ in range(steps):
            if v == 0:
                return state
            viol = [t for t in range(len(remaining_tests))
                    if sum(state[j] for j in test_members[t]) != targets[t]]
            t = rng.choice(viol)
            members = test_members[t]
            cnt = sum(state[j] for j in members)
            cands = ([j for j in members if state[j] == 1] if cnt > targets[t]
                     else [j for j in members if state[j] == 0])
            best_j, best_v = None, None
            for j in cands:
                state[j] ^= 1
                nv = total_violation(state)
                state[j] ^= 1
                if best_v is None or nv < best_v:
                    best_v, best_j = nv, j
            state[best_j] ^= 1
            v = total_violation(state)
        if v == 0:
            return state
    return None


def _connected_components(active_list, remaining_tests):
    """Partition active agents into connected components, where two agents are
    connected iff they co-occur in some remaining test. Tests never cross
    components (all agents of a test are mutually connected), so the posterior
    factorizes across components and each can be solved independently. Returns a
    list of sorted agent lists."""
    parent = {i: i for i in active_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pool_mask, r in remaining_tests:
        members = [i for i in active_list if pool_mask >> i & 1]
        for k in range(1, len(members)):
            union(members[0], members[k])

    comps = {}
    for i in active_list:
        comps.setdefault(find(i), []).append(i)
    return [sorted(c) for c in comps.values()]


def _component_tests(comp, remaining_tests):
    """Tests that live entirely inside this component, as (members, r) pairs."""
    comp_set = set(comp)
    tests = []
    for pool_mask, r in remaining_tests:
        members = [a for a in comp if pool_mask >> a & 1]
        if members:
            tests.append((members, r))
    return tests


def _exact_component_marginals(comp, tests, p):
    """Exact P(agent active | tests) by enumerating the component's 2^|comp|
    assignments consistent with its exact-count tests. Cost 2^|comp| * k,
    independent of n. Raises ValueError if the component is infeasible."""
    m = len(comp)
    pos = {a: b for b, a in enumerate(comp)}
    test_masks = []
    for members, r in tests:
        bm = 0
        for a in members:
            bm |= (1 << pos[a])
        test_masks.append((bm, r))
    pa = [p[a] for a in comp]
    qa = [1.0 - p[a] for a in comp]

    total = 0.0
    active_w = [0.0] * m
    for assign in range(1 << m):
        ok = True
        for bm, r in test_masks:
            if (assign & bm).bit_count() != r:
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for b in range(m):
            w *= pa[b] if (assign >> b & 1) else qa[b]
        total += w
        bits = assign
        while bits:
            lsb = bits & -bits
            b = lsb.bit_length() - 1
            active_w[b] += w
            bits ^= lsb
    if total <= 0.0:
        raise ValueError("infeasible component in gibbs_update")
    return {comp[b]: active_w[b] / total for b in range(m)}


def _propose_alternating_move(comp, tests, agent_tests, state, rng, max_steps,
                              count_preserving_only=False):
    """Propose a Markov-basis move on the exact-count fiber: a kernel vector
    delta in {-1,0,+1} with A·delta = 0 that is applicable to `state`
    (flips 0->1 where +1, 1->0 where -1). Built as a randomized alternating
    path/cycle in the agent-test incidence: flipping one agent unbalances its
    tests, each repaired by flipping a partner the other way, propagating until
    every test is balanced again. Crucially these moves CAN change the total
    active count (e.g. (+1,-1,+1) on {0,1}=1,{1,2}=1), which single-site/swap
    moves cannot — restoring ergodicity across count levels.

    The proposal is ASYMMETRIC: the eligible-partner counts along the path
    differ between `state` and the proposed state, so Metropolis acceptance
    needs the Hastings factor q(z'->z)/q(z->z'). The reverse of this exact
    path (same start agent, same deterministic test order, flip directions
    inverted) is the unique mirror proposal from the new state, so
    q(rev)/q(fwd) = prod(|elig_fwd_k|) / prod(|elig_rev_k|), where the reverse
    eligible sets are computed under the flipped assignment with the same
    already-in-path exclusions.

    Returns (move, log_hastings) with move a dict {agent: new_value} and
    log_hastings = log q(rev) - log q(fwd), or None if the path dead-ends
    within the step budget."""
    a0 = rng.choice(comp)
    delta = {a0: 1 - 2 * state[a0]}          # +1 (0->1) or -1 (1->0)
    pending = {}                              # test_idx -> net imbalance to undo
    for ti in agent_tests[a0]:
        pending[ti] = pending.get(ti, 0) + delta[a0]

    path = []                                 # (test_idx, want, n_eligible_fwd)
    steps = 0
    while any(v != 0 for v in pending.values()):
        steps += 1
        if steps > max_steps:
            return None
        ti = next(t for t, v in pending.items() if v != 0)
        imbalance = pending[ti]
        want = -1 if imbalance > 0 else +1   # direction to flip a partner
        members, r = tests[ti]
        elig = [a for a in members
                if a not in delta
                and ((want == -1 and state[a] == 1)
                     or (want == +1 and state[a] == 0))]
        if not elig:
            return None                      # dead end -> reject
        a = rng.choice(elig)
        path.append((ti, a, want, len(elig)))
        delta[a] = want
        for tj in agent_tests[a]:
            pending[tj] = pending.get(tj, 0) + want

    move = {a: state[a] + d for a, d in delta.items()}

    if count_preserving_only and sum(delta.values()) != 0:
        return None       # ablation kernel: reject count-changing moves

    # Mirror-path replay: eligible counts of the reverse proposal from `move`.
    # At reverse step k the excluded set is the same {a0, chosen_1..chosen_{k-1}}
    # and every agent still keeps its post-move value (agents in delta are
    # flipped; agents outside delta are unchanged).
    log_corr = 0.0
    seen = {a0}
    for ti, chosen, want, n_fwd in path:
        members, r = tests[ti]
        rev_want = -want
        n_rev = 0
        for a in members:
            if a in seen:
                continue
            sa = move[a] if a in move else state[a]
            if (rev_want == -1 and sa == 1) or (rev_want == +1 and sa == 0):
                n_rev += 1
        seen.add(chosen)
        log_corr += math.log(n_fwd) - math.log(n_rev)

    return move, log_corr


def _deduce_from_history(history, n):
    """Deterministic-deduction preprocessing shared by gibbs_update and
    posterior_draws: propagate r=0 (all clearancey) and r=|pool| (all active)
    deductions to a fixed point. Returns (confirmed_clearancey,
    confirmed_active, remaining_tests) where remaining_tests holds the
    reduced (eff_pool, eff_r) pairs still carrying uncertainty."""
    confirmed_clearancey = set()
    confirmed_active = set()

    remaining_tests = [(pool_mask, r) for pool_mask, r in history]
    changed = True
    while changed:
        changed = False
        new_tests = []
        for pool_mask, r in remaining_tests:
            # Remove confirmed agents from this test
            eff_pool = pool_mask
            eff_r = r
            for i in confirmed_clearancey:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i)
            for i in confirmed_active:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i)
                    eff_r -= 1

            pool_size = popcount(eff_pool)

            if eff_r == 0 and eff_pool != 0:
                # All remaining in pool are clearancey
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_clearancey:
                        confirmed_clearancey.add(i)
                        changed = True
            elif eff_r == pool_size and pool_size > 0:
                # All remaining in pool are active
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_active:
                        confirmed_active.add(i)
                        changed = True
            elif eff_r > 0 and pool_size > 0:
                new_tests.append((eff_pool, eff_r))
        remaining_tests = new_tests

    return confirmed_clearancey, confirmed_active, remaining_tests


def _alternating_move_component_marginals(comp, tests, p, rng,
                                          num_iterations, burn_in,
                                          window_size, tolerance,
                                          count_preserving_only=False,
                                          stats=None, collect=None):
    """Metropolis-Hastings generator over a connected component using
    alternating-path Markov moves (count-changing, hence ergodic across count
    levels) with prior-ratio acceptance times the mirror-path Hastings factor
    (the proposal is asymmetric; without the correction the stationary
    distribution is biased — audit 2026-07-06). Used only when a single
    connected component is too large for exact enumeration (rare). Validated
    against exact enumeration on small forced instances.

    With count_preserving_only the proposal is restricted to its
    count-preserving subset (the old swap generator); this is the ablation
    baseline and is deliberately non-ergodic across count levels. When a list
    is passed as ``stats`` one dict per MCMC-solved component is appended with
    the visited count-level histogram and the proposal/acceptance tallies.
    When a list is passed as ``collect``, one full assignment snapshot
    (dict {agent: 0/1}) is appended per kept iteration; consecutive snapshots
    are autocorrelated (thinning = one sweep of move attempts)."""
    agent_tests = {a: [] for a in comp}
    for ti, (members, r) in enumerate(tests):
        for a in members:
            agent_tests[a].append(ti)

    # Seed a valid state by min-conflicts (reuse the existing helper).
    remaining = [( _mask(members), r) for members, r in tests]
    state = _find_valid_state(remaining, comp, p, rng)
    if state is None:
        # Last resort: exact (capped) — the component must be feasible.
        return _exact_component_marginals(comp, tests, p)

    max_steps = 6 * len(comp) + 12
    active_counts = {a: 0 for a in comp}
    total_draws = 0
    prev = None
    proposed = 0
    accepted = 0
    count_hist = {}
    for it in range(num_iterations):
        # several move attempts per sweep to decorrelate
        for _ in range(max(len(comp), 5)):
            proposal = _propose_alternating_move(
                comp, tests, agent_tests, state, rng, max_steps,
                count_preserving_only=count_preserving_only)
            if proposal is None:
                continue
            proposed += 1
            move, log_ratio = proposal   # start from the Hastings factor
            ok = True
            for a, nv in move.items():
                ov = state[a]
                if nv == ov:
                    continue
                num = p[a] if nv == 1 else (1.0 - p[a])
                den = p[a] if ov == 1 else (1.0 - p[a])
                if num <= 0.0:
                    ok = False
                    break
                log_ratio += math.log(num) - math.log(den) if den > 0 else 50.0
            if ok and (log_ratio >= 0 or rng.random() < math.exp(log_ratio)):
                state.update(move)
                accepted += 1
        if it >= burn_in:
            lvl = 0
            for a in comp:
                if state[a] == 1:
                    active_counts[a] += 1
                    lvl += 1
            count_hist[lvl] = count_hist.get(lvl, 0) + 1
            total_draws += 1
            if collect is not None:
                collect.append(dict(state))
            if total_draws % window_size == 0:
                cur = {a: active_counts[a] / total_draws for a in comp}
                if prev is not None and max(abs(cur[a] - prev[a])
                                            for a in comp) < tolerance:
                    break
                prev = cur

    if total_draws == 0:
        return _exact_component_marginals(comp, tests, p)
    if stats is not None:
        stats.append({"comp": list(comp),
                      "tests": [(list(members), r) for members, r in tests],
                      "count_hist": count_hist,
                      "proposed": proposed, "accepted": accepted,
                      "draws": total_draws})
    return {a: active_counts[a] / total_draws for a in comp}


def _mask(agents):
    m = 0
    for a in agents:
        m |= (1 << a)
    return m


def gibbs_update(p, history, n, num_iterations=1000, burn_in=200,
                 window_size=50, tolerance=1e-4, seed=None,
                 count_preserving_only=False, mcmc_stats=None):
    """Approximate posterior marginals via Gibbs drawing (MCMC).

    Adapted from Appendix A.2 of "Dynamic Welfare-Maximizing Adaptive Group Counting"
    for augmented tests where
    each test returns the exact count r = |t ∩ Z| of active in the pool.

    The algorithm:
      1. Preprocessing: deterministic deductions (r=0 → all clearancey,
         r=|pool| → all active), with constraint propagation.
      2. Decompose the remaining agents into connected components
         (agents are connected iff they co-occur in a remaining test);
         the posterior factorizes across components.
      3. Solve every component with at most EXACT_ACTIVE_THRESHOLD
         agents exactly by enumerating its 2^|component| assignments.
      4. Only a larger single component falls back to MCMC: Metropolis-
         Hastings on the exact-count fiber via alternating-path moves —
         which can change the total active count, unlike the removed
         single-site/swap/block generator — with the mirror-path
         Hastings correction for the asymmetric proposal.
      5. After burn-in, collect draws and estimate marginals as
         empirical frequencies, with a rolling-window convergence stop.

    Parameters
    ----------
    p : list[float]
        Prior latent-state probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Full test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.
    num_iterations : int
        Maximum number of Gibbs iterations.
    burn_in : int
        Number of initial iterations to discard.
    window_size : int
        Rolling window size for convergence monitoring.
    tolerance : float
        Convergence threshold: stop if max change in marginals < tolerance.
    seed : int or None
        Random seed for reproducibility.
    count_preserving_only : bool
        Ablation switch. When True, the MCMC proposal is restricted to its
        count-preserving subset (the removed swap generator), which is
        non-ergodic across count levels and biased on multi-level fibers.
        Default False reproduces the ergodic sampler exactly.
    mcmc_stats : list or None
        Optional collector. When a list is passed, one dict per MCMC-solved
        component is appended: {"comp", "tests", "count_hist", "proposed",
        "accepted", "draws"}. Components solved by exact enumeration append
        nothing.

    Returns
    -------
    list[float]
        Posterior latent-state probabilities P(Z_i=1 | h_k).

    Notes
    -----
    Complexity: O(n * |history| * num_iterations).
    Scales to n~50+. For small n, results approximate those from
    bayesian_update_by_counting (which is exact but O(2^n)); for very
    small active subproblems, this function uses exact counting directly.
    """
    import random as _random

    if not history:
        return list(p)

    rng = _random.Random(seed)

    # ---- Step 1: Preprocessing — deterministic deductions ----
    confirmed_clearancey, confirmed_active, remaining_tests = \
        _deduce_from_history(history, n)

    # Build posterior for confirmed agents
    posterior = list(p)
    for i in confirmed_clearancey:
        posterior[i] = 0.0
    for i in confirmed_active:
        posterior[i] = 1.0

    # Identify active agents (those in at least one remaining test)
    active_set = set()
    for pool_mask, r in remaining_tests:
        for i in range(n):
            if pool_mask >> i & 1:
                active_set.add(i)

    if not active_set:
        return posterior

    active_list = sorted(active_set)

    # ---- Step 2: Decompose into connected components and solve each ----
    # Agents are connected iff they co-occur in a remaining test, so the
    # posterior factorizes across components. We solve each component EXACTLY by
    # enumerating its 2^|component| feasible assignments — this is both exact and
    # ergodicity-free, and because the cap applies PER COMPONENT it covers far
    # larger active sets than the old monolithic 2^|active| shortcut. Only a
    # single connected component larger than the cap (rare in practice) falls
    # back to MCMC, and that MCMC uses alternating-path Markov moves that can
    # change the total active count — the moves the old single-site/swap/block
    # generator lacked, which is why it got stuck on overlapping exact-count pools.
    components = _connected_components(active_list, remaining_tests)
    for comp in components:
        comp_tests = _component_tests(comp, remaining_tests)
        if len(comp) <= EXACT_ACTIVE_THRESHOLD:
            marg = _exact_component_marginals(comp, comp_tests, p)
        else:
            marg = _alternating_move_component_marginals(
                comp, comp_tests, p, rng,
                num_iterations=num_iterations, burn_in=burn_in,
                window_size=window_size, tolerance=tolerance,
                count_preserving_only=count_preserving_only,
                stats=mcmc_stats)
        for a, val in marg.items():
            posterior[a] = val

    return posterior


def _component_assignment_draws(comp, tests, p, rng, num_draws):
    """Draw num_draws iid assignments for one connected component by
    enumerating its consistent assignments with prior weights (the same
    enumeration _exact_component_marginals performs) and sampling the
    resulting categorical. Returns a list of GLOBAL bitmasks (bits only on
    this component's agents). Raises ValueError if infeasible."""
    m = len(comp)
    pos = {a: b for b, a in enumerate(comp)}
    test_masks = []
    for members, r in tests:
        bm = 0
        for a in members:
            bm |= (1 << pos[a])
        test_masks.append((bm, r))
    pa = [p[a] for a in comp]
    qa = [1.0 - p[a] for a in comp]

    assigns, weights = [], []
    for assign in range(1 << m):
        ok = True
        for bm, r in test_masks:
            if (assign & bm).bit_count() != r:
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for b in range(m):
            w *= pa[b] if (assign >> b & 1) else qa[b]
        if w > 0.0:
            assigns.append(assign)
            weights.append(w)
    if not assigns:
        raise ValueError("infeasible component in posterior_draws")

    out = []
    for assign in rng.choices(assigns, weights=weights, k=num_draws):
        gm = 0
        bits = assign
        while bits:
            lsb = bits & -bits
            gm |= (1 << comp[lsb.bit_length() - 1])
            bits ^= lsb
        out.append(gm)
    return out


def posterior_draws(p, history, n, num_draws=1000, seed=None,
                    num_iterations=1000, burn_in=200, window_size=50,
                    tolerance=1e-4):
    """Draw full latent-state profiles Z ~ P(Z | history).

    Reuses the gibbs_update pipeline stages: deterministic deductions fix the
    forced bits; each residual connected component with at most
    EXACT_ACTIVE_THRESHOLD agents is enumerated and sampled iid from its
    exact categorical; an oversized component falls back to the alternating-
    move MCMC (one thinned assignment per kept iteration, recycled with a
    warning if it converges before num_draws); agents in no residual test are
    iid Bernoulli(p_i).

    Draws are iid exact for instances whose components all fit the
    enumeration threshold; MCMC components introduce autocorrelation.

    Returns
    -------
    list[int]
        num_draws full n-bit masks (bit i set = agent i active).
    """
    import random as _random

    rng = _random.Random(seed)

    if history:
        confirmed_clearancey, confirmed_active, remaining_tests = \
            _deduce_from_history(history, n)
    else:
        confirmed_clearancey, confirmed_active, remaining_tests = \
            set(), set(), []

    forced_mask = 0
    for i in confirmed_active:
        forced_mask |= (1 << i)
    draws = [forced_mask] * num_draws

    active_set = set()
    for pool_mask, _ in remaining_tests:
        for i in range(n):
            if pool_mask >> i & 1:
                active_set.add(i)

    components = (_connected_components(sorted(active_set), remaining_tests)
                  if active_set else [])

    for comp in components:
        comp_tests = _component_tests(comp, remaining_tests)
        if len(comp) <= EXACT_ACTIVE_THRESHOLD:
            comp_draws = _component_assignment_draws(
                comp, comp_tests, p, rng, num_draws)
        else:
            collected = []
            _alternating_move_component_marginals(
                comp, comp_tests, p, rng,
                num_iterations=num_iterations, burn_in=burn_in,
                window_size=window_size, tolerance=tolerance,
                collect=collected)
            if not collected:
                # Mirror of the marginal path's last resort: the MCMC could
                # not seed/keep a state, so enumerate (the component must be
                # feasible for the history to have happened).
                comp_draws = _component_assignment_draws(
                    comp, comp_tests, p, rng, num_draws)
            else:
                if len(collected) < num_draws:
                    warnings.warn(
                        f"posterior_draws: componente MCMC produjo "
                        f"{len(collected)} draws < {num_draws}; se reciclan "
                        "muestras (autocorrelacion adicional)",
                        RuntimeWarning, stacklevel=2)
                comp_draws = []
                for k in range(num_draws):
                    st = collected[k % len(collected)]
                    gm = 0
                    for a, v in st.items():
                        if v:
                            gm |= (1 << a)
                    comp_draws.append(gm)
        for k in range(num_draws):
            draws[k] |= comp_draws[k]

    # Unconstrained agents (in no residual test): iid Bernoulli(p_i).
    constrained = confirmed_clearancey | confirmed_active | active_set
    for i in range(n):
        if i in constrained:
            continue
        pi = p[i]
        if pi <= 0.0:
            continue
        bit = 1 << i
        for k in range(num_draws):
            if rng.random() < pi:
                draws[k] |= bit

    return draws


def estimate_p_from_history(history, n, prior_p=None):
    """Posterior marginals from a prior guess and the observed history.

    Pese al nombre historico, aqui no hay suavizado Beta: la funcion devuelve
    bayesian_update_by_counting sobre prior_p (0.5 por omision) y, sin
    historia, el prior tal cual. (Los parametros Beta muertos que sugerian
    otra cosa se eliminaron en la limpieza 2026-07.)

    Parameters
    ----------
    history : tuple of (pool_mask, result) pairs
        Test history.
    n : int
        Population size.
    prior_p : list[float] or None
        Prior guess for latent-state probabilities (default: 0.5 each).

    Returns
    -------
    list[float]
        Posterior latent-state probabilities.
    """
    p_prior = list(prior_p) if prior_p else [0.5] * n
    if not history:
        return p_prior
    return bayesian_update_by_counting(p_prior, history, n)
