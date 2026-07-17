"""Certificados: cotas superiores computables para certificar politicas
tratables contra el optimo incalculable (linea D3).

Dos cotas:

U_PI (informacion perfecta / hindsight): un adversario que conoce el perfil
latente Z limpia a las B*G personas limpias de mayor utilidad. Domina a toda
politica adaptada, es exacta por enumeracion en n chico y Monte Carlo a
cualquier escala, pero se afloja cuando B*G se acerca al numero de limpios.

U_pen (relajacion de informacion con penalizacion, Brown-Smith-Sun): al
adversario se le cobra, en cada paso, la diferencia entre el valor estimado
tras ver el resultado real y su esperanza bajo el posterior verdadero
(diferencia de martingala de una funcion de valor aproximada V-hat). Bajo la
filtracion natural la penalizacion tiene media cero, asi que

    OPT <= E_Z [ max_{t_1..t_B} ( welfare(Z, pools) - sum_t pi_t ) ] = U_pen

para CUALQUIER V-hat; una V-hat informativa aprieta la cota. El problema
interno se resuelve exacto (DP sobre historias) — restringirlo invalidaria la
cota. Costo: exponencial en n y B; usar en n <= ~6.

El welfare replica la semantica del simulador (simulator.apply_dapts):
se acredita u_i cuando i pertenece a algun pool testeado con resultado 0.
"""

import math
import random

from augmented.core import test_result, popcount
from augmented.bayesian import (exact_pool_pmf, bayesian_update_by_counting,
                                gibbs_update)
from augmented.greedy import greedy_myopic_expected_utility, EXACT_PMF_MAX_N
from augmented.vhat import get as _get_vhat

# Umbral para el posterior expuesto a las V-hat: exacto (2^n) en n chico, Gibbs
# corregido (por componentes) por encima. Hace que la primitiva de marginales
# escale, de modo que una V-hat basada en marginales (p.ej. umax) corre a n=50
# igual que en producción, mientras que una V-hat que enumera el soporte
# CONJUNTO (2^n) no. Los certificados con puntaje viven en n<=6, así que su
# U_pen se computa siempre por la rama exacta y no cambia. La frontera es LA
# MISMA que la de los pesos de rama exactos (greedy.EXACT_PMF_MAX_N): una
# sola fuente de verdad para "hasta donde es exacta la inferencia".
_EXACT_POSTERIOR_MAX_N = EXACT_PMF_MAX_N


# -------------------------------------------------------------------
# U_PI — cota de informacion perfecta
# -------------------------------------------------------------------

def _pi_welfare(z_mask, u, n, cap):
    """Con Z conocido, lo mejor posible: las top-cap utilidades limpias."""
    clean = sorted((u[i] for i in range(n) if not (z_mask >> i & 1)),
                   reverse=True)
    return sum(clean[:cap])


def u_pi_exact(p, u, B, G):
    """E_Z[top B*G utilidades limpias], por enumeracion (n <= ~20)."""
    n = len(p)
    if B * G >= n:
        # Regimen saturado: _pi_welfare(z) = suma de TODAS las utilidades
        # limpias para todo z, asi que por linealidad U_PI = sum u_i (1-p_i),
        # exacta y O(n). (El MC de antes estimaba esta constante.)
        return sum(u[i] * (1.0 - p[i]) for i in range(n))
    q = [1.0 - pi for pi in p]
    cap = B * G
    total = 0.0
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else q[i]
        if w > 0.0:
            total += w * _pi_welfare(z, u, n, cap)
    return total


def u_pi_mc(p, u, B, G, num_samples=100000, seed=0):
    """Version Monte Carlo de U_PI para n grande."""
    n = len(p)
    if B * G >= n:
        # Regimen saturado: _pi_welfare(z) = suma de TODAS las utilidades
        # limpias para todo z, asi que por linealidad U_PI = sum u_i (1-p_i),
        # exacta y O(n). (El MC de antes estimaba esta constante.)
        return sum(u[i] * (1.0 - p[i]) for i in range(n))
    cap = B * G
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(num_samples):
        z = 0
        for i in range(n):
            if rng.random() < p[i]:
                z |= (1 << i)
        acc += _pi_welfare(z, u, n, cap)
    return acc / num_samples


# -------------------------------------------------------------------
# U_pen — cota penalizada
# -------------------------------------------------------------------

def _all_pools(n, G):
    return [m for m in range(1, 1 << n) if popcount(m) <= G]


class _PenaltyEngine:
    """Contexto para las V-hat del registro (augmented/vhat.py) y PMFs
    predictivas, con caches por historia (compartidas entre perfiles Z: todo
    depende solo de la historia, no de Z). La construccion de la penalizacion
    y el problema interno viven AQUI (referencia intocable); las V-hat viven
    en vhat.py (superficie editable). Esa frontera de archivos es la frontera
    del teorema de validez."""

    def __init__(self, p, u, n, G, v_hat):
        self.p = p
        self.u = u
        self.n = n
        self.G = G
        self.kind = v_hat
        self._fn = _get_vhat(v_hat)
        self._vhat_cache = {}
        self._pmf_cache = {}
        self._post_cache = {}

    # --- primitivas cacheadas expuestas a las V-hat (ctx.*) ---

    def posterior(self, h_fs):
        """Marginales posteriores P(Z_i=1 | h), cacheadas. Exactas en n chico,
        Gibbs corregido (por componentes) por encima de _EXACT_POSTERIOR_MAX_N.
        Es la primitiva escalable: no enumera el soporte conjunto."""
        val = self._post_cache.get(h_fs)
        if val is None:
            if self.n <= _EXACT_POSTERIOR_MAX_N:
                val = bayesian_update_by_counting(self.p, tuple(h_fs), self.n)
            else:
                val = gibbs_update(self.p, tuple(h_fs), self.n, seed=0)
            self._post_cache[h_fs] = val
        return val

    def cleared_mask(self, h_fs):
        """Bitmask de individuos acreditados (en algun pool con r=0)."""
        cleared = 0
        for pool, r in h_fs:
            if r == 0:
                cleared |= pool
        return cleared

    def greedy_value(self, p, u, budget):
        """EU del greedy miope secuencial con presupuesto `budget`."""
        if budget <= 0:
            return 0.0
        return greedy_myopic_expected_utility(p, u, budget, self.G)

    # --- evaluacion de la V-hat registrada ---

    def vhat(self, h_fs, remaining):
        key = (h_fs, remaining)
        val = self._vhat_cache.get(key)
        if val is None:
            val = self._fn(self, h_fs, remaining)
            self._vhat_cache[key] = val
        return val

    def pmf(self, h_fs, pool):
        key = (h_fs, pool)
        val = self._pmf_cache.get(key)
        if val is None:
            val = exact_pool_pmf(self.p, tuple(h_fs), pool, self.n)
            self._pmf_cache[key] = val
        return val

    def penalty(self, h_fs, pool, r_obs, scale, remaining):
        """pi = scale * ( V(h+(a,r_obs)) - E_{r~P(.|h,a)}[V(h+(a,r))] ),
        con V evaluada en el estado SIGUIENTE (presupuesto `remaining`)."""
        if self.kind == "zero" or scale == 0.0:
            return 0.0
        pmf = self.pmf(h_fs, pool)
        exp_v = 0.0
        for r, pr in enumerate(pmf):
            if pr > 0.0:
                exp_v += pr * self.vhat(h_fs | {(pool, r)}, remaining)
        v_obs = self.vhat(h_fs | {(pool, r_obs)}, remaining)
        return scale * (v_obs - exp_v)


def _cleared_welfare(h_fs, u, n):
    """Semantica del simulador: u_i acreditada si i esta en un pool con r=0."""
    cleared = 0
    for pool, r in h_fs:
        if r == 0:
            cleared |= pool
    return sum(u[i] for i in range(n) if cleared >> i & 1)


def u_pen_exact(p, u, B, G, v_hat="umax", scales=(0.5, 1.0, 2.0)):
    """Cota penalizada exacta.

    Para cada escala c, U_pen(c) = E_Z[max interno con penalizacion c*pi] es
    una cota valida; se devuelve min_c U_pen(c). El minimo se toma sobre los
    AGREGADOS (el minimo por-perfil no seria una cota valida).

    v_hat: nombre de una funcion registrada en augmented/vhat.py ("zero"
    recupera U_PI; "umax" es el potencial posterior; "greedy" el
    valor-a-futuro del greedy; "research" el slot de busqueda del harness).
    Cualquier V-hat registrada da cota valida; solo la tightness varia.
    """
    n = len(p)
    q = [1.0 - pi for pi in p]
    pools = _all_pools(n, G)
    engine = _PenaltyEngine(p, u, n, G, v_hat)
    if v_hat == "zero":
        scales = (0.0,)

    weights = []
    for z in range(1 << n):
        w = 1.0
        for i in range(n):
            w *= p[i] if (z >> i & 1) else q[i]
        if w > 0.0:
            weights.append((z, w))

    best = None
    for scale in scales:
        total = 0.0
        for z, w in weights:
            memo = {}

            def inner(h_fs, t):
                if t == B:
                    return _cleared_welfare(h_fs, u, n)
                key = (h_fs, t)
                val = memo.get(key)
                if val is not None:
                    return val
                best_a = -math.inf
                for a in pools:
                    r_obs = test_result(a, z)
                    pi_t = engine.penalty(h_fs, a, r_obs, scale,
                                          remaining=B - t - 1)
                    nxt = inner(h_fs | {(a, r_obs)}, t + 1)
                    cand = nxt - pi_t
                    if cand > best_a:
                        best_a = cand
                memo[key] = best_a
                return best_a

            total += w * inner(frozenset(), 0)
        if best is None or total < best:
            best = total
    return best
