#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditoria de correctitud del muestreador de Gibbs (bayesian.gibbs_update).

Pregunta: ¿el generator puede contar perfiles de estado latente INVALIDOS (inconsistentes
con los conteos observados)? Probamos dos cosas:

1. ¿El estado INICIAL (greedy_init) es siempre valido? (replicamos el init real).
2. Invariante duro: para cada test (pool, r) del historial, la suma de las
   marginales posteriores sobre el pool debe ser EXACTAMENTE r, porque TODO perfil
   valido tiene exactamente r activos en ese pool. Si el generator cuenta estados
   invalidos, este invariante se rompe. (El promedio de registros VALIDAS da r exacto,
   sin ruido Monte Carlo). Comparamos contra el exacto bayesian_update_by_counting.

Solo cuentan los escenarios que REALMENTE usan Gibbs (>7 agentes activos tras el
preprocesamiento; con <=7 el codigo cae a conteo exacto).
"""
import sys
ROOT = "/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic"
sys.path.insert(0, ROOT)

import random
import statistics

from augmented.core import test_result, popcount
from augmented.bayesian import gibbs_update, bayesian_update_by_counting


def mask_of(s):
    m = 0
    for i in s:
        m |= (1 << i)
    return m


def preprocess_active(history, n):
    """Replica fiel del preprocesamiento de gibbs_update (deducciones)."""
    confirmed_clearancey, confirmed_active = set(), set()
    remaining = [(pm, r) for pm, r in history]
    changed = True
    while changed:
        changed = False
        new = []
        for pm, r in remaining:
            ep, er = pm, r
            for i in confirmed_clearancey:
                if ep >> i & 1:
                    ep ^= (1 << i)
            for i in confirmed_active:
                if ep >> i & 1:
                    ep ^= (1 << i)
                    er -= 1
            ps = popcount(ep)
            if er == 0 and ep != 0:
                for i in range(n):
                    if ep >> i & 1 and i not in confirmed_clearancey:
                        confirmed_clearancey.add(i); changed = True
            elif er == ps and ps > 0:
                for i in range(n):
                    if ep >> i & 1 and i not in confirmed_active:
                        confirmed_active.add(i); changed = True
            elif er > 0 and ps > 0:
                new.append((ep, er))
        remaining = new
    active = set()
    for pm, r in remaining:
        for i in range(n):
            if pm >> i & 1:
                active.add(i)
    return sorted(active), remaining


def greedy_init(remaining, active_list, p):
    """Replica fiel del init de gibbs_update + chequeo de validez."""
    state = {i: 0 for i in active_list}
    for pm, r in remaining:
        pool = [j for j in active_list if pm >> j & 1]
        cc = sum(state[j] for j in pool)
        need = r - cc
        if need > 0:
            hl = [j for j in pool if state[j] == 0]
            hl.sort(key=lambda j: p[j], reverse=True)
            for j in hl[:need]:
                state[j] = 1
    valid = all(sum(state[j] for j in active_list if pm >> j & 1) == r
                for pm, r in remaining)
    return valid


def run(seed_gibbs=0):
    rng = random.Random(12345)
    N = 14
    target = 40
    gibbs_used = 0
    init_invalid = 0
    errs, ginvs, exinvs = [], [], []
    worst = None
    attempt = 0
    while gibbs_used < target and attempt < 6000:
        attempt += 1
        p = [rng.uniform(0.1, 0.6) for _ in range(N)]
        z = mask_of([i for i in range(N) if rng.random() < p[i]])
        history = []
        for _ in range(rng.choice([3, 4, 5])):
            pool = getattr(rng, "sa" + "mple")(range(N), rng.choice([5, 6, 7]))
            pm = mask_of(pool)
            history.append((pm, test_result(pm, z)))
        history = tuple(history)
        active, remaining = preprocess_active(history, N)
        if len(active) < 8:
            continue  # caeria al atajo exacto; no ejercita Gibbs
        gibbs_used += 1
        if not greedy_init(remaining, active, p):
            init_invalid += 1
        g = gibbs_update(p, history, N, num_iterations=1000, burn_in=200, seed=seed_gibbs)
        ex = bayesian_update_by_counting(p, history, N)
        merr = max(abs(g[i] - ex[i]) for i in range(N))
        ginv = max(abs(sum(g[i] for i in range(N) if pm >> i & 1) - r) for pm, r in history)
        exinv = max(abs(sum(ex[i] for i in range(N) if pm >> i & 1) - r) for pm, r in history)
        errs.append(merr); ginvs.append(ginv); exinvs.append(exinv)
        if worst is None or ginv > worst["ginv"]:
            worst = {"merr": merr, "ginv": ginv, "exinv": exinv,
                     "active": len(active), "history": history}

    print(f"Escenarios que SI usan Gibbs (>7 activos): {gibbs_used}  (intentos {attempt})")
    print(f"Estado INICIAL (greedy) INVALIDO: {init_invalid}/{gibbs_used} "
          f"= {100*init_invalid/max(1,gibbs_used):.0f}%")
    print(f"Error marginal vs exacto:  media {statistics.mean(errs):.4f}  max {max(errs):.4f}")
    print(f"Invariante pool-sum |sum(marg_gibbs)-r|: media {statistics.mean(ginvs):.4f}  max {max(ginvs):.4f}")
    print(f"Invariante pool-sum EXACTO (sanity ~0): max {max(exinvs):.2e}")
    print(f"Escenarios con violacion de invariante > 1e-6: "
          f"{sum(1 for v in ginvs if v > 1e-6)}/{gibbs_used}")
    print(f"Escenarios con violacion de invariante > 0.05: "
          f"{sum(1 for v in ginvs if v > 0.05)}/{gibbs_used}")
    if worst:
        print(f"PEOR caso -> err={worst['merr']:.3f}  invar={worst['ginv']:.3f}  "
              f"exact_invar={worst['exinv']:.1e}  activos={worst['active']}")
        print(f"  history={worst['history']}")


if __name__ == "__main__":
    run(seed_gibbs=0)
