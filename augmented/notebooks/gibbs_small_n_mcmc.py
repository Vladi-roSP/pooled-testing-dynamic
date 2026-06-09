#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""¿El MCMC de Gibbs funciona para n chico SIN caer al atajo exacto?

Forzamos el camino MCMC bajando el umbral del atajo exacto a 0 (en runtime, sin
tocar bayesian.py) y comparamos contra el conteo exacto (bayesian_update_by_counting).
"""
import sys
ROOT = "/Users/hectorbecerrilvillamil/Desktop/PooledTesting/pooled-testing-dynamic"
sys.path.insert(0, ROOT)

import random
import statistics
import augmented.bayesian as B
from augmented.bayesian import gibbs_update, bayesian_update_by_counting
from augmented.core import test_result

# --- Forzar MCMC: desactivar el atajo exacto y el fallback exacto ---
B.EXACT_ACTIVE_THRESHOLD = 0     # nunca enumerar exacto por conjunto activo chico
B.EXACT_ACTIVE_FALLBACK_CAP = 0  # ni siquiera como respaldo: vemos el MCMC crudo


def mask(idxs):
    m = 0
    for i in idxs:
        m |= (1 << i)
    return m


def err_and_inv(p, history, n, iters=4000):
    g = gibbs_update(p, history, n, num_iterations=iters, burn_in=iters // 5, seed=0)
    ex = bayesian_update_by_counting(p, history, n)
    err = max(abs(g[i] - ex[i]) for i in range(n))
    inv = max(abs(sum(g[i] for i in range(n) if pm >> i & 1) - r) for pm, r in history)
    return err, inv, g, ex


print("=" * 72)
print("CASO A: un solo pool, n=4, {0,1,2,3} con r=2 (espacio valido conexo)")
print("=" * 72)
pA = [0.2, 0.4, 0.3, 0.5]
hA = ((mask([0, 1, 2, 3]), 2),)
err, inv, g, ex = err_and_inv(pA, hA, 4)
print(f"  MCMC : {[round(x,3) for x in g]}")
print(f"  exacto:{[round(x,3) for x in ex]}")
print(f"  error max = {err:.4f}   invariante = {inv:.2e}")

print()
print("=" * 72)
print("CASO B: dos pools solapados, n=5, {0,1,2}=1 y {2,3,4}=1")
print("  (perfiles validos con DISTINTO total de infectados -> no ergodico)")
print("=" * 72)
pB = [0.3, 0.3, 0.3, 0.3, 0.3]
hB = ((mask([0, 1, 2]), 1), (mask([2, 3, 4]), 1))
for iters in [1000, 5000, 20000]:
    err, inv, g, ex = err_and_inv(pB, hB, 5, iters=iters)
    print(f"  iters={iters:6d}  error max = {err:.4f}  invariante = {inv:.2e}")
print(f"  MCMC (20k): {[round(x,3) for x in g]}")
print(f"  exacto    : {[round(x,3) for x in ex]}")

print()
print("=" * 72)
print("BARRIDO ALEATORIO de n chico (4,5,6), historias factibles")
print("=" * 72)
rng = random.Random(0)
for n in [4, 5, 6]:
    errs, invs = [], []
    for _ in range(60):
        p = [rng.uniform(0.1, 0.6) for _ in range(n)]
        z = mask([i for i in range(n) if rng.random() < p[i]])
        ntests = rng.choice([2, 3])
        pools = [mask(rng.sample(range(n), rng.choice([2, 3, min(4, n)]))) for _ in range(ntests)]
        history = tuple((pm, test_result(pm, z)) for pm in pools)
        # solo casos que de verdad dejan algo que muestrear
        e, iv, _, _ = err_and_inv(p, history, n, iters=4000)
        errs.append(e); invs.append(iv)
    peor = max(errs)
    malos = sum(1 for e in errs if e > 0.02)
    print(f"  n={n}: error medio {statistics.mean(errs):.3f} | peor {peor:.3f} | "
          f"casos con error>0.02: {malos}/{len(errs)} | peor invariante {max(invs):.1e}")

print()
print("Conclusion: el MCMC es exacto donde el espacio de perfiles validos es CONEXO")
print("bajo intercambios sana<->infectada (un pool, o pools que no parten el total).")
print("Cuando dos pools solapados admiten perfiles con DISTINTO total de infectados,")
print("el MCMC queda atrapado en un componente y sesga -- aunque n sea chico.")
print("El invariante (suma=r) se respeta siempre: nunca cuenta perfiles invalidos.")
