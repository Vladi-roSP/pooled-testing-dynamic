#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression test: gibbs_update must NOT count infection profiles that are
inconsistent with the observed exact-count history.

Two checks per scenario (only scenarios that actually exercise the MCMC path,
i.e. > 7 active agents after preprocessing, but small enough that the exact
brute force is the ground truth):

  1. Accuracy:  max_i |gibbs_update[i] - bayesian_update_by_counting[i]| < TOL
  2. Hard invariant: for every (pool, r) in history,
        sum_i in pool  posterior[i]  ==  r   (exactly, up to MC/float noise)
     because EVERY valid profile has exactly r infected in that pool. The exact
     method satisfies this to ~1e-15; a sampler that counts invalid profiles
     violates it.

Run:  /Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_gibbs_validity.py
Exit code 0 = PASS, 1 = FAIL.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from augmented.core import test_result, popcount
from augmented.bayesian import gibbs_update, bayesian_update_by_counting

TOL = 1e-6


def _mask_of(s):
    m = 0
    for i in s:
        m |= (1 << i)
    return m


def _active_after_preprocessing(history, n):
    confirmed_healthy, confirmed_infected = set(), set()
    remaining = [(pm, r) for pm, r in history]
    changed = True
    while changed:
        changed = False
        new = []
        for pm, r in remaining:
            ep, er = pm, r
            for i in confirmed_healthy:
                if ep >> i & 1:
                    ep ^= (1 << i)
            for i in confirmed_infected:
                if ep >> i & 1:
                    ep ^= (1 << i); er -= 1
            ps = popcount(ep)
            if er == 0 and ep != 0:
                for i in range(n):
                    if ep >> i & 1 and i not in confirmed_healthy:
                        confirmed_healthy.add(i); changed = True
            elif er == ps and ps > 0:
                for i in range(n):
                    if ep >> i & 1 and i not in confirmed_infected:
                        confirmed_infected.add(i); changed = True
            elif er > 0 and ps > 0:
                new.append((ep, er))
        remaining = new
    active = set()
    for pm, r in remaining:
        for i in range(n):
            if pm >> i & 1:
                active.add(i)
    return len(active)


def _make_scenarios(n=14, n_scen=25, min_active=8, max_active=16, seed=2024):
    rng = random.Random(seed)
    scen = []
    attempts = 0
    while len(scen) < n_scen and attempts < 8000:
        attempts += 1
        p = [rng.uniform(0.1, 0.6) for _ in range(n)]
        z = _mask_of([i for i in range(n) if rng.random() < p[i]])
        history = []
        for _ in range(rng.choice([3, 4, 5])):
            pool = rng.sample(range(n), rng.choice([5, 6, 7]))
            pm = _mask_of(pool)
            history.append((pm, test_result(pm, z)))
        history = tuple(history)
        a = _active_after_preprocessing(history, n)
        if min_active <= a <= max_active:
            scen.append((p, history))
    return scen


def main():
    scen = _make_scenarios()
    n = 14
    fails = 0
    worst_err = 0.0
    worst_inv = 0.0
    for k, (p, history) in enumerate(scen):
        g = gibbs_update(p, history, n, seed=0)
        ex = bayesian_update_by_counting(p, history, n)
        err = max(abs(g[i] - ex[i]) for i in range(n))
        inv = max(abs(sum(g[i] for i in range(n) if pm >> i & 1) - r)
                  for pm, r in history)
        worst_err = max(worst_err, err)
        worst_inv = max(worst_inv, inv)
        if err > TOL or inv > TOL:
            fails += 1

    print(f"Scenarios (>7 active, exact path): {len(scen)}")
    print(f"worst |gibbs - exact|        = {worst_err:.4f}  (TOL {TOL})")
    print(f"worst pool-sum invariant viol = {worst_inv:.4f}  (TOL {TOL})")
    if fails:
        print(f"FAIL: {fails}/{len(scen)} scenarios exceed tolerance "
              f"(gibbs counts invalid profiles / wrong marginals)")
        return 1
    print(f"PASS: all {len(scen)} scenarios within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
