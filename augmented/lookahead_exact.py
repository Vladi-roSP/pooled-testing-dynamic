"""Lookahead exacto de profundidad 1 sobre el conjunto de perfiles consistentes.

La politica: en cada paso, cada pool candidato se puntua con
E_r[ valor de la continuacion greedy exacta ] — anticipa UN paso y asume
greedy (exacto) despues — y se juega el argmax, re-planificando en cada paso
(horizonte rodante). Tanto los pesos de rama como el scoring viven en la
representacion de frozenset-de-perfiles-consistentes (cero producto de
marginales, cero updates secuenciales): este es el lookahead con el cableado
correcto, contraparte del legacy `greedy._lookahead_best_pool` (PB +
secuencial) cuya degradacion media la tabla vieja 99/40/16.

Costo O(2^n * |pools| * B * estados alcanzables) — para n <= 8.
"""

from augmented.core import all_pools, test_result
from augmented.independence_gap import _prior_weights_indep, _exact_best_pool


def exact_lookahead_expected_utility(p, u, B, G):
    """Valor esperado exacto de la politica lookahead-1 exactamente cableada.

    En B=2 anticipar el unico paso futuro es la optimizacion completa, asi
    que coincide con solve_optimal_dapts; en B=1 degenera al miope exacto.
    """
    n = len(p)
    w = _prior_weights_indep(p, n)
    pools = all_pools(n, G, include_empty=False)

    def _util(cleared):
        return sum(u[i] for i in range(n) if cleared >> i & 1)

    greedy_memo = {}

    def greedy_value(k, remaining, cleared):
        """Continuacion greedy miope exacta (misma recursion que
        independence_gap.exact_greedy_myopic_expected_utility)."""
        key = (k, remaining, cleared)
        if key in greedy_memo:
            return greedy_memo[key]
        if k == B or not remaining:
            greedy_memo[key] = _util(cleared)
            return greedy_memo[key]
        pool, score = _exact_best_pool(remaining, cleared, u, pools, w, n)
        if pool == 0 or score <= 0.0:
            greedy_memo[key] = _util(cleared)
            return greedy_memo[key]
        total_mass = sum(w[z] for z in remaining)
        buckets = {}
        for z in remaining:
            buckets.setdefault(test_result(pool, z), []).append(z)
        ev = 0.0
        for r, zs in buckets.items():
            mass_r = sum(w[z] for z in zs)
            new_cleared = cleared | pool if r == 0 else cleared
            ev += (mass_r / total_mass) * greedy_value(
                k + 1, frozenset(zs), new_cleared)
        greedy_memo[key] = ev
        return ev

    la_memo = {}

    def lookahead_value(k, remaining, cleared):
        key = (k, remaining, cleared)
        if key in la_memo:
            return la_memo[key]
        if k == B or not remaining:
            la_memo[key] = _util(cleared)
            return la_memo[key]

        total_mass = sum(w[z] for z in remaining)
        # Seleccion: puntua cada pool con la continuacion greedy exacta.
        # "No hacer nada" (gastar el test) es el punto de partida del argmax,
        # espejo de la opcion nothing del legacy _lookahead_best_pool.
        best_pool = 0
        best_score = greedy_value(k + 1, remaining, cleared)
        for pool in pools:
            buckets = {}
            for z in remaining:
                buckets.setdefault(test_result(pool, z), []).append(z)
            score = 0.0
            for r, zs in buckets.items():
                mass_r = sum(w[z] for z in zs)
                new_cleared = cleared | pool if r == 0 else cleared
                score += (mass_r / total_mass) * greedy_value(
                    k + 1, frozenset(zs), new_cleared)
            if score > best_score + 1e-15:
                best_score, best_pool = score, pool

        if best_pool == 0:
            # Gastar el test y seguir con la misma politica.
            la_memo[key] = lookahead_value(k + 1, remaining, cleared)
            return la_memo[key]

        # Jugar el pool elegido; la continuacion REAL re-planifica con
        # lookahead (no con la greedy usada para puntuar).
        buckets = {}
        for z in remaining:
            buckets.setdefault(test_result(best_pool, z), []).append(z)
        ev = 0.0
        for r, zs in buckets.items():
            mass_r = sum(w[z] for z in zs)
            new_cleared = cleared | best_pool if r == 0 else cleared
            ev += (mass_r / total_mass) * lookahead_value(
                k + 1, frozenset(zs), new_cleared)
        la_memo[key] = ev
        return ev

    return lookahead_value(0, frozenset(range(1 << n)), 0)
