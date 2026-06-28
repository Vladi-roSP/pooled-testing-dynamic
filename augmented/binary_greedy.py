"""
Greedy del régimen CLÁSICO (binario), para medir la separación vs el aumentado.

El régimen aumentado observa el conteo exacto r = |t ∩ Z|; el clásico solo observa
si el pool salió positivo (r>0) o negativo (r=0). La regla de selección es la misma
greedy miope (`_myopic_best_pool`): solo difiere la ACTUALIZACIÓN del posterior.
Así, comparar `greedy_myopic_simulate` (aumentado, usa r) contra
`greedy_binary_simulate` (binario, usa solo el signo) aísla el valor del conteo bajo
la misma heurística — la separación que persigue Francisco, computable a escala
porque ambas usan updates secuenciales O(|pool|), no enumeración 2^n.
"""

from augmented.core import indices_from_mask, test_result
from augmented.greedy import _myopic_best_pool


def binary_update_single_test(p, pool_mask, positive, n):
    """Actualización bayesiana del régimen binario tras un test.

    positive=False (negativo, r=0): todos los del pool quedan sanos -> p_i = 0.
    positive=True  (positivo, r>0): bajo independencia,
        P(Z_i=1 | al menos uno infectado en t) = p_i / (1 - prod_{j in t}(1-p_j)),
    que sube la probabilidad de cada miembro pero NO dice cuántos hay (eso es lo que
    el conteo sí revela).
    """
    idx = indices_from_mask(pool_mask, n)
    if not idx:
        return list(p)
    post = list(p)
    if not positive:
        for i in idx:
            post[i] = 0.0
        return post
    prob_all_healthy = 1.0
    for i in idx:
        prob_all_healthy *= (1.0 - p[i])
    denom = 1.0 - prob_all_healthy
    if denom > 1e-15:
        for i in idx:
            post[i] = min(1.0, p[i] / denom)
    return post


def greedy_binary_simulate(p, u, B, G, z_mask, pool_selector=None):
    """Greedy miope en el régimen binario sobre un perfil fijo z_mask.

    Misma selección que el aumentado; observa solo positivo/negativo y actualiza con
    `binary_update_single_test`. Devuelve (history, cleared_mask, utility), con
    history en pares (pool, 1|0) donde 1 = positivo.
    """
    n = len(p)
    cur = list(p)
    cleared = 0
    history = ()
    select = pool_selector if pool_selector is not None else _myopic_best_pool
    for _ in range(B):
        pool = select(cur, u, G, n, cleared)
        if pool == 0:
            break
        positive = test_result(pool, z_mask) > 0
        history = history + ((pool, 1 if positive else 0),)
        if not positive:
            cleared |= pool
        cur = binary_update_single_test(cur, pool, positive, n)
    utility = sum(u[i] for i in indices_from_mask(cleared, n))
    return history, cleared, utility


def greedy_binary_counting_simulate(p, u, B, G, z_mask, pool_selector=None):
    """Greedy binario con posterior de TODA la historia (full-history) vía
    `binary_update_by_counting`. Es el rival justo del greedy aumentado con conteo
    (`greedy_myopic_counting_simulate`): ambos hacen TODAS las deducciones que su
    observación permite, así que la diferencia es puramente conteo vs binario. Solo
    para n donde 2^n es enumerable (n <= ~18)."""
    n = len(p)
    cleared = 0
    history = ()
    select = pool_selector if pool_selector is not None else _myopic_best_pool
    for _ in range(B):
        cur = binary_update_by_counting(p, history, n) if history else list(p)
        pool = select(cur, u, G, n, cleared)
        if pool == 0:
            break
        positive = test_result(pool, z_mask) > 0
        history = history + ((pool, positive),)
        if not positive:
            cleared |= pool
    utility = sum(u[i] for i in indices_from_mask(cleared, n))
    return history, cleared, utility


def binary_update_by_counting(p, history_binary, n):
    """Posterior EXACTO del régimen binario (para validación en n chico).

    history_binary: pares (pool_mask, positive_bool). Enumera los 2^n perfiles y
    conserva los consistentes con cada signo observado (test_result==0 <-> negativo).
    """
    q = [1.0 - x for x in p]
    total = 0.0
    inf_w = [0.0] * n
    for z in range(1 << n):
        ok = True
        for pool, positive in history_binary:
            neg = (test_result(pool, z) == 0)
            if neg == bool(positive):   # negativo observado debe casar con test==0
                ok = False
                break
        if not ok:
            continue
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else q[i]
        total += w
        bits = z
        while bits:
            lsb = bits & -bits
            inf_w[lsb.bit_length() - 1] += w
            bits ^= lsb
    if total <= 0:
        raise ValueError("historia binaria infeasible")
    return [inf_w[i] / total for i in range(n)]
