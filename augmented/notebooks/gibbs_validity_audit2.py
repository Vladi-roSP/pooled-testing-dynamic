#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditoria #2 (instrumentada): prueba DIRECTA de cuantas muestras invalidas
cuenta el Gibbs y que tan mal mezcla, mas un barrido realista con pools G=3.

Copia FIEL de bayesian.gibbs_update con instrumentacion: en cada muestra
(iteration >= burn_in) registra si el estado es valido y guarda el estado para
contar cuantos estados DISTINTOS visita (mezcla).
"""
import sys
ROOT = "/Users/hectorbecerrilvillamil/Desktop/PooledTesting/pooled-testing-dynamic"
sys.path.insert(0, ROOT)

import random as _random
import statistics

from augmented.core import test_result, popcount
from augmented.bayesian import bayesian_update_by_counting


def mask_of(s):
    m = 0
    for i in s:
        m |= (1 << i)
    return m


def gibbs_instrumented(p, history, n, num_iterations=1000, burn_in=200,
                       window_size=50, tolerance=1e-4, seed=0):
    """Copia fiel de gibbs_update + contadores de validez/mezcla."""
    rng = _random.Random(seed)
    confirmed_healthy, confirmed_infected = set(), set()
    remaining_tests = [(pm, r) for pm, r in history]
    changed = True
    while changed:
        changed = False
        new_tests = []
        for pool_mask, r in remaining_tests:
            eff_pool, eff_r = pool_mask, r
            for i in confirmed_healthy:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i)
            for i in confirmed_infected:
                if eff_pool >> i & 1:
                    eff_pool ^= (1 << i); eff_r -= 1
            pool_size = popcount(eff_pool)
            if eff_r == 0 and eff_pool != 0:
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_healthy:
                        confirmed_healthy.add(i); changed = True
            elif eff_r == pool_size and pool_size > 0:
                for i in range(n):
                    if eff_pool >> i & 1 and i not in confirmed_infected:
                        confirmed_infected.add(i); changed = True
            elif eff_r > 0 and pool_size > 0:
                new_tests.append((eff_pool, eff_r))
        remaining_tests = new_tests

    posterior = list(p)
    for i in confirmed_healthy:
        posterior[i] = 0.0
    for i in confirmed_infected:
        posterior[i] = 1.0

    active_set = set()
    for pool_mask, r in remaining_tests:
        for i in range(n):
            if pool_mask >> i & 1:
                active_set.add(i)
    if not active_set:
        return posterior, None  # resuelto deterministicamente
    active_list = sorted(active_set)
    if len(active_list) <= 7:
        return bayesian_update_by_counting(p, history, n), None  # atajo exacto

    agent_tests = {i: [] for i in active_list}
    for idx, (pool_mask, r) in enumerate(remaining_tests):
        for i in active_list:
            if pool_mask >> i & 1:
                agent_tests[i].append(idx)

    state = {i: 0 for i in active_list}
    for pool_mask, r in remaining_tests:
        pool_agents = [j for j in active_list if pool_mask >> j & 1]
        cc = sum(state[j] for j in pool_agents)
        need = r - cc
        if need > 0:
            hl = [j for j in pool_agents if state[j] == 0]
            hl.sort(key=lambda j: p[j], reverse=True)
            for j in hl[:need]:
                state[j] = 1

    def _state_valid():
        for pm, rv in remaining_tests:
            if sum(state[j] for j in active_list if pm >> j & 1) != rv:
                return False
        return True

    def _count_infected(ti):
        pm = remaining_tests[ti][0]
        return sum(state[j] for j in active_list if pm >> j & 1)

    healthy_counts = {i: 0 for i in active_list}
    total_samples = 0
    prev_marginals = None
    n_invalid_samples = 0
    distinct = set()

    for iteration in range(num_iterations):
        order = list(active_list); rng.shuffle(order)
        for i in order:
            infected_ok = healthy_ok = True
            for test_idx in agent_tests[i]:
                pool_mask, r = remaining_tests[test_idx]
                other = sum(1 for j in active_list
                            if j != i and (pool_mask >> j & 1) and state[j] == 1)
                if other + 1 != r:
                    infected_ok = False
                if other != r:
                    healthy_ok = False
            if infected_ok and healthy_ok:
                state[i] = 1 if rng.random() < p[i] else 0
            elif infected_ok:
                state[i] = 1
            elif healthy_ok:
                state[i] = 0
        for test_idx, (pool_mask, r) in enumerate(remaining_tests):
            inf = [j for j in active_list if (pool_mask >> j & 1) and state[j] == 1]
            hlt = [j for j in active_list if (pool_mask >> j & 1) and state[j] == 0]
            if not inf or not hlt:
                continue
            a, b = rng.choice(inf), rng.choice(hlt)
            state[a], state[b] = 0, 1
            ok = all(_count_infected(t) == remaining_tests[t][1]
                     for t in set(agent_tests[a]) | set(agent_tests[b]))
            if ok:
                pn, po = p[b] * (1 - p[a]), p[a] * (1 - p[b])
                acc = min(1.0, pn / po) if po > 0 else 1.0
                if rng.random() >= acc:
                    state[a], state[b] = 1, 0
            else:
                state[a], state[b] = 1, 0
        for _ in range(max(len(active_list), 5)):
            inf = [j for j in active_list if state[j] == 1]
            hlt = [j for j in active_list if state[j] == 0]
            if not inf or not hlt:
                break
            a, b = rng.choice(inf), rng.choice(hlt)
            state[a], state[b] = 0, 1
            if _state_valid():
                pn, po = p[b] * (1 - p[a]), p[a] * (1 - p[b])
                acc = min(1.0, pn / po) if po > 0 else 1.0
                if rng.random() >= acc:
                    state[a], state[b] = 1, 0
            else:
                state[a], state[b] = 1, 0
        if iteration >= burn_in:
            if not _state_valid():
                n_invalid_samples += 1
            distinct.add(tuple(state[i] for i in active_list))
            for i in active_list:
                if state[i] == 0:
                    healthy_counts[i] += 1
            total_samples += 1
            if total_samples % window_size == 0:
                cur = {i: 1.0 - healthy_counts[i] / total_samples for i in active_list}
                if prev_marginals is not None:
                    if max(abs(cur[i] - prev_marginals[i]) for i in active_list) < tolerance:
                        break
                prev_marginals = cur

    if total_samples > 0:
        for i in active_list:
            posterior[i] = 1.0 - healthy_counts[i] / total_samples
    frac_invalid = n_invalid_samples / max(1, total_samples)
    return posterior, {"frac_invalid": frac_invalid, "distinct": len(distinct),
                       "samples": total_samples, "active": len(active_list)}


def sweep(label, N, pool_sizes, ntests_choices, target=30):
    rng = _random.Random(777)
    used = 0; attempt = 0
    fracs, distincts, errs, hit = [], [], [], 0
    while used < target and attempt < 8000:
        attempt += 1
        p = [rng.uniform(0.1, 0.6) for _ in range(N)]
        z = mask_of([i for i in range(N) if rng.random() < p[i]])
        history = []
        for _ in range(rng.choice(ntests_choices)):
            pool = rng.sample(range(N), rng.choice(pool_sizes))
            pm = mask_of(pool)
            history.append((pm, test_result(pm, z)))
        history = tuple(history)
        post, info = gibbs_instrumented(p, history, N, seed=0)
        if info is None:
            continue  # atajo exacto / resuelto
        used += 1
        ex = bayesian_update_by_counting(p, history, N)
        err = max(abs(post[i] - ex[i]) for i in range(N))
        fracs.append(info["frac_invalid"]); distincts.append(info["distinct"]); errs.append(err)
    print(f"\n[{label}]  escenarios que usan Gibbs: {used} (de {attempt} intentos)")
    if used:
        print(f"  fraccion de MUESTRAS INVALIDAS contadas: media {statistics.mean(fracs):.2f}  max {max(fracs):.2f}")
        print(f"  estados DISTINTOS visitados (mezcla):    media {statistics.mean(distincts):.1f}  min {min(distincts)}")
        print(f"  error marginal vs exacto:                media {statistics.mean(errs):.3f}  max {max(errs):.3f}")
        print(f"  escenarios con >10% muestras invalidas:  {sum(1 for f in fracs if f>0.1)}/{used}")


if __name__ == "__main__":
    # Pools grandes (G=5-7): el caso de la auditoria #1
    sweep("pools grandes G=5-7, n=14", 14, [5, 6, 7], [3, 4, 5], target=30)
    # Pools realistas G=3 (como muchos experimentos), n=16 con muchos tests
    sweep("pools realistas G=3, n=16", 16, [3], [6, 7, 8, 9], target=30)
