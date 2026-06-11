"""
Bayesian posterior update for augmented pooled tests.

Given prior infection probabilities p = (p_1, ..., p_n) and a test history
H_k = ((t_1, r_1), ..., (t_k, r_k)), compute posterior probabilities
q'_i = P(individual i is healthy | H_k) for each individual.

In the idealized augmented model, testing pool t yields r = |t ∩ Z|
(the exact count of infected in the pool).
"""

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
    """Update infection probabilities after one augmented test.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
    pool_mask : int
        Bitmask of the tested pool t.
    r : int
        Observed result r = |t ∩ Z| (count of infected in pool).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities p'_i = P(Z_i=1 | r, t).

    Math
    ----
    For i NOT in t: test gives no info, so p'_i = p_i.

    For i IN t, by Bayes:
        p'_i = P(Z_i=1 | r) = P(r | Z_i=1) * p_i / P(r)

    where (letting S = t \\ {i}):
        P(r | Z_i=1) = P(exactly r-1 infected in S)   [Poisson-Binomial]
        P(r | Z_i=0) = P(exactly r   infected in S)   [Poisson-Binomial]
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

        # PMF of number of infected among others
        pmf = _poisson_binomial_pmf(others_p)

        # P(r | Z_i = 1) = P(r-1 infected among others)
        p_r_given_1 = pmf[r - 1] if r >= 1 else 0.0
        # P(r | Z_i = 0) = P(r infected among others)
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
        Prior infection probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities after all tests in history.
    """
    current_p = list(p)
    for pool_mask, r in history:
        current_p = bayesian_update_single_test(current_p, pool_mask, r, n)
    return current_p


def bayesian_update_by_counting(p, history, n):
    """Compute posterior P(Z_i=1 | h_k) by counting over all consistent worlds.

    This is the "by counting" approach: enumerate all 2^n infection profiles,
    keep those consistent with the full test history, and compute posteriors
    as weighted proportions.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
    history : tuple of (pool_mask, result) pairs
        Full test history H_k = ((t_1, r_1), ..., (t_k, r_k)).
    n : int
        Population size.

    Returns
    -------
    list[float]
        Posterior infection probabilities P(Z_i=1 | h_k).

    Notes
    -----
    Complexity: O(2^n * k) where k = len(history).
    Feasible for n <= ~20. This is the EXACT joint posterior. It coincides
    with sequential single-test updates (`bayesian_update`) only when the
    tested pools are pairwise disjoint; with OVERLAPPING pools the two differ,
    because the sequential update treats marginals as independent and so misses
    the cross-test deductions that counting captures (e.g. tests {0,1}=1 and
    {1,2}=0 force individual 0 infected — counting sees it, sequential does not).
    """
    if not history:
        return list(p)

    q = [1.0 - pi for pi in p]
    num_profiles = 1 << n

    # Accumulate: weighted count of consistent profiles, and per-individual
    # weighted count of consistent profiles where Z_i = 1
    total_weight = 0.0
    infected_weight = [0.0] * n

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
        # Add weight to each infected individual in this profile
        bits = z_mask
        while bits:
            lsb = bits & -bits
            i = lsb.bit_length() - 1
            infected_weight[i] += w
            bits ^= lsb

    # Compute posteriors. total_weight==0 means NO infection profile is
    # consistent with the history (or all consistent profiles have zero prior
    # probability). Returning the prior here would silently hide a bug, so raise.
    if total_weight <= 0.0:
        raise ValueError(
            "bayesian_update_by_counting: infeasible history — no infection "
            "profile is consistent with it (or all have zero prior mass). "
            "Refusing to silently return the prior."
        )
    posterior = list(p)
    for i in range(n):
        posterior[i] = infected_weight[i] / total_weight

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
    posterior MARGINALS is NOT the true distribution of r_t (it can put positive
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


# Subproblemas con <= EXACT_ACTIVE_THRESHOLD agentes activos (tras el
# preprocesamiento) se resuelven por enumeracion EXACTA sobre el conjunto activo
# (costo 2^|activos|, independiente de n). Cubre todas las escalas reales del
# proyecto (el DP exacto topa en n<=14). Por encima de eso se usa MCMC con guard
# de validez; si el MCMC no junta muestras validas, hay un fallback exacto capado.
EXACT_ACTIVE_THRESHOLD = 16
EXACT_ACTIVE_FALLBACK_CAP = 22


def _exact_active_marginals(p, remaining_tests, active_list, posterior, n):
    """Marginales posteriores EXACTAS enumerando solo el conjunto ACTIVO.

    Tras el preprocesamiento, los agentes confirmados ya estan fijos en
    ``posterior`` (0/1) y los no-activos sin restriccion conservan su prior. La
    posterior conjunta de los activos es la enumeracion ponderada por el prior
    sobre las 2^|activos| asignaciones que satisfacen ``remaining_tests`` (cuyos
    pools/conteos ya estan reducidos al conjunto activo). Costo O(2^A * k),
    independiente de n, y numericamente identico a la enumeracion completa 2^n.
    """
    A = active_list
    m = len(A)
    pos = {agent: b for b, agent in enumerate(A)}
    test_masks = []
    for eff_pool, eff_r in remaining_tests:
        bm = 0
        for agent in A:
            if eff_pool >> agent & 1:
                bm |= (1 << pos[agent])
        test_masks.append((bm, eff_r))
    pa = [p[agent] for agent in A]
    qa = [1.0 - p[agent] for agent in A]

    total_w = 0.0
    infected_w = [0.0] * m
    for assign in range(1 << m):
        ok = True
        for bm, eff_r in test_masks:
            if (assign & bm).bit_count() != eff_r:
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for b in range(m):
            w *= pa[b] if (assign >> b & 1) else qa[b]
        total_w += w
        bits = assign
        while bits:
            lsb = bits & -bits
            b = lsb.bit_length() - 1
            infected_w[b] += w
            bits ^= lsb
    if total_w > 0:
        for b, agent in enumerate(A):
            posterior[agent] = infected_w[b] / total_w
    return posterior


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


def gibbs_update(p, history, n, num_iterations=1000, burn_in=200,
                 window_size=50, tolerance=1e-4, seed=None):
    """Approximate posterior marginals via Gibbs sampling (MCMC).

    Adapted from Appendix A.2 of "Dynamic Welfare-Maximizing Pooled Testing"
    for augmented tests where
    each test returns the exact count r = |t ∩ Z| of infected in the pool.

    The algorithm:
      1. Preprocessing: deterministic deductions (r=0 → all healthy,
         r=|pool| → all infected), with constraint propagation.
      2. Initialize state vector by sampling from priors.
      3. For very small active subproblems, fall back to exact counting
         to avoid mixing pathologies in disconnected feasible regions.
      4. Gibbs iterations: for each agent in random order, compute the
         conditional P(X_i | X_{-i}) by checking consistency with all
         test constraints, then sample or force deterministically.
      5. Add Metropolis-Hastings swap proposals, both within individual
         test pools and across the full active set, to improve mixing.
      6. After burn-in, collect samples and estimate marginals as
         empirical frequencies.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities (length n).
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

    Returns
    -------
    list[float]
        Posterior infection probabilities P(Z_i=1 | h_k).

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
    confirmed_healthy = set()
    confirmed_infected = set()

    remaining_tests = [(pool_mask, r) for pool_mask, r in history]
    changed = True
    while changed:
        changed = False
        new_tests = []
        for pool_mask, r in remaining_tests:
            # Remove confirmed agents from this test
            eff_pool = pool_mask
            eff_r = r
            for i in confirmed_healthy:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i)
            for i in confirmed_infected:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i)
                    eff_r -= 1

            pool_size = popcount(eff_pool)

            if eff_r == 0 and eff_pool != 0:
                # All remaining in pool are healthy
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_healthy:
                        confirmed_healthy.add(i)
                        changed = True
            elif eff_r == pool_size and pool_size > 0:
                # All remaining in pool are infected
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_infected:
                        confirmed_infected.add(i)
                        changed = True
            elif eff_r > 0 and pool_size > 0:
                new_tests.append((eff_pool, eff_r))
        remaining_tests = new_tests

    # Build posterior for confirmed agents
    posterior = list(p)
    for i in confirmed_healthy:
        posterior[i] = 0.0
    for i in confirmed_infected:
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

    # Enumeracion EXACTA sobre el conjunto activo (costo 2^|activos|,
    # independiente de n). Correcto y barato para todas las escalas reales.
    # Reemplaza el atajo anterior `<=7 -> 2^n`, que (a) no cubria subproblemas
    # acoplados mas grandes y (b) era O(2^n), por lo que colgaba con n grande aun
    # con pocos activos.
    if len(active_list) <= EXACT_ACTIVE_THRESHOLD:
        return _exact_active_marginals(p, remaining_tests, active_list, posterior, n)

    # For each active agent, precompute which tests involve them
    agent_tests = {i: [] for i in active_list}
    for idx, (pool_mask, r) in enumerate(remaining_tests):
        for i in active_list:
            if pool_mask >> i & 1:
                agent_tests[i].append(idx)

    # ---- Step 2: Initialize state to a VALID (consistent) profile ----
    # El init greedy anterior podia arrancar INVALIDO cuando los pools se
    # solapan, y un arranque invalido no se reparaba de forma confiable y
    # contaminaba el estimado. Sembramos con busqueda de minimos conflictos para
    # que la cadena empiece consistente.
    state = _find_valid_state(remaining_tests, active_list, p, rng)
    if state is None:
        # Fallback: asignacion greedy (puede ser invalida) — el guard de mas
        # abajo asegura que solo se cuenten muestras validas.
        state = {i: 0 for i in active_list}
        for pool_mask, r in remaining_tests:
            pool_agents = [j for j in active_list if pool_mask >> j & 1]
            needed = r - sum(state[j] for j in pool_agents)
            if needed > 0:
                healthy_in_pool = [j for j in pool_agents if state[j] == 0]
                healthy_in_pool.sort(key=lambda j: p[j], reverse=True)
                for j in healthy_in_pool[:needed]:
                    state[j] = 1

    # Helper: check if current state satisfies all test constraints
    def _state_valid():
        for pm, r_val in remaining_tests:
            cnt = sum(state[j] for j in active_list if pm >> j & 1)
            if cnt != r_val:
                return False
        return True

    # Helper: count infected in a test pool
    def _count_infected(test_idx):
        pm = remaining_tests[test_idx][0]
        return sum(state[j] for j in active_list if pm >> j & 1)

    # ---- Step 3: Gibbs iterations with swap moves ----
    healthy_counts = {i: 0 for i in active_list}
    total_samples = 0
    prev_marginals = None

    for iteration in range(num_iterations):
        # --- Standard Gibbs sweep ---
        order = list(active_list)
        rng.shuffle(order)

        for i in order:
            infected_ok = True
            healthy_ok = True

            for test_idx in agent_tests[i]:
                pool_mask, r = remaining_tests[test_idx]
                other_infected = 0
                for j in active_list:
                    if j != i and (pool_mask >> j & 1) and state[j] == 1:
                        other_infected += 1

                if other_infected + 1 != r:
                    infected_ok = False
                if other_infected != r:
                    healthy_ok = False

            if infected_ok and healthy_ok:
                state[i] = 1 if rng.random() < p[i] else 0
            elif infected_ok:
                state[i] = 1
            elif healthy_ok:
                state[i] = 0
            # else: neither consistent — keep current state

        # --- Swap moves (Metropolis-Hastings) ---
        # For each test with 0 < r < |pool|, propose swapping an infected
        # and a healthy member. This ensures ergodicity for exact-count
        # constraints where standard Gibbs gets stuck.
        for test_idx, (pool_mask, r) in enumerate(remaining_tests):
            infected_in_pool = [j for j in active_list
                                if (pool_mask >> j & 1) and state[j] == 1]
            healthy_in_pool = [j for j in active_list
                               if (pool_mask >> j & 1) and state[j] == 0]

            if not infected_in_pool or not healthy_in_pool:
                continue

            i_inf = rng.choice(infected_in_pool)
            i_hlt = rng.choice(healthy_in_pool)

            # Temporarily swap
            state[i_inf], state[i_hlt] = 0, 1

            # Check all tests involving either agent
            swap_valid = True
            for tidx in set(agent_tests[i_inf]) | set(agent_tests[i_hlt]):
                if _count_infected(tidx) != remaining_tests[tidx][1]:
                    swap_valid = False
                    break

            if swap_valid:
                # Metropolis-Hastings acceptance ratio based on priors
                p_new = p[i_hlt] * (1.0 - p[i_inf])
                p_old = p[i_inf] * (1.0 - p[i_hlt])
                acceptance = min(1.0, p_new / p_old) if p_old > 0 else 1.0

                if rng.random() >= acceptance:
                    state[i_inf], state[i_hlt] = 1, 0  # reject
            else:
                state[i_inf], state[i_hlt] = 1, 0  # revert invalid swap

        # --- Global pairwise block moves ---
        # These proposals are not restricted to a single test pool, so they
        # can move across broader feasible regions when pools overlap.
        n_block_moves = max(len(active_list), 5)
        for _ in range(n_block_moves):
            inf_agents = [j for j in active_list if state[j] == 1]
            hlt_agents = [j for j in active_list if state[j] == 0]

            if not inf_agents or not hlt_agents:
                break

            i_inf = rng.choice(inf_agents)
            i_hlt = rng.choice(hlt_agents)
            state[i_inf], state[i_hlt] = 0, 1

            if _state_valid():
                p_new = p[i_hlt] * (1.0 - p[i_inf])
                p_old = p[i_inf] * (1.0 - p[i_hlt])
                acceptance = min(1.0, p_new / p_old) if p_old > 0 else 1.0

                if rng.random() >= acceptance:
                    state[i_inf], state[i_hlt] = 1, 0  # reject
            else:
                state[i_inf], state[i_hlt] = 1, 0  # revert invalid swap

        # ---- Step 4: Collect samples after burn-in (SOLO estados validos) ----
        # Guard: nunca contar un estado que viole los conteos observados.
        # total_samples y la ventana de convergencia avanzan solo en validas.
        if iteration >= burn_in and _state_valid():
            for i in active_list:
                if state[i] == 0:
                    healthy_counts[i] += 1
            total_samples += 1

            # Convergence check
            if total_samples > 0 and total_samples % window_size == 0:
                current_marginals = {i: 1.0 - healthy_counts[i] / total_samples
                                     for i in active_list}
                if prev_marginals is not None:
                    max_diff = max(abs(current_marginals[i] - prev_marginals[i])
                                  for i in active_list)
                    if max_diff < tolerance:
                        break
                prev_marginals = current_marginals

    # Compute posteriors from the collected VALID samples.
    if total_samples > 0:
        for i in active_list:
            posterior[i] = 1.0 - healthy_counts[i] / total_samples
    elif len(active_list) <= EXACT_ACTIVE_FALLBACK_CAP:
        # No se junto ninguna muestra valida (p.ej. mala mezcla). Fallback exacto
        # CAPADO: garantiza correctitud sin arriesgar un cuelgue 2^grande.
        posterior = _exact_active_marginals(p, remaining_tests, active_list,
                                            posterior, n)
    # else: se dejan las confirmaciones deterministas + prior para los activos
    #       (aproximado, pero de costo acotado; solo para conjuntos activos enormes).

    return posterior


def estimate_p_from_history(history, n, prior_p=None, prior_strength=1.0):
    """Estimate infection probabilities from observed test data.

    Uses a Bayesian approach with a Beta prior. If prior_p is given,
    uses Beta(prior_strength * p_i, prior_strength * (1 - p_i)) as prior.
    Otherwise uses uniform Beta(1, 1).

    Parameters
    ----------
    history : tuple of (pool_mask, result) pairs
        Test history.
    n : int
        Population size.
    prior_p : list[float] or None
        Prior guess for infection probabilities.
    prior_strength : float
        Strength of the prior (pseudo-count scale).

    Returns
    -------
    list[float]
        Estimated infection probabilities.
    """
    if prior_p is not None:
        alpha = [prior_strength * pi for pi in prior_p]
        beta = [prior_strength * (1.0 - pi) for pi in prior_p]
    else:
        alpha = [1.0] * n
        beta = [1.0] * n

    # For each consistent world, accumulate posterior mass
    if not history:
        return [a / (a + b) for a, b in zip(alpha, beta)]

    q_prior = [1.0] * n
    if prior_p:
        q_prior = [1.0 - pi for pi in prior_p]
    else:
        q_prior = [0.5] * n

    p_prior = prior_p if prior_p else [0.5] * n

    # Use counting-based posterior
    return bayesian_update_by_counting(p_prior, history, n)
