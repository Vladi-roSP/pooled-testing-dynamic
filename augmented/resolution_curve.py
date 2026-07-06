"""Curva de resolución: utilidad óptima vs profundidad de truncamiento del
conteo. cap=1 es binario, cap>=G es conteo completo (§5 del spec de diseño).
"""

from augmented.solver import solve_optimal_dapts


def cap_chain(G):
    """Caps con sentido: [1, 2, ..., G]. Un pool tiene <= G miembros, así que
    cap >= G equivale al conteo completo."""
    return list(range(1, G + 1))


def resolution_curve(p, u, B, G, caps=None):
    """Devuelve [{'cap': k, 'value': U_k}, ...] a lo largo de la cadena de caps.
    El cap máximo (== G) se resuelve como conteo completo (cap=None)."""
    if caps is None:
        caps = cap_chain(G)
    out = []
    for cap in caps:
        eff_cap = None if cap >= G else cap  # cap>=G == conteo completo
        value, _ = solve_optimal_dapts(p, u, B, G, cap=eff_cap)
        out.append({"cap": cap, "value": value})
    return out


def fraction_captured(curve):
    """Fracción del beneficio del conteo capturada por cada cap:
    (U_k - U_bin) / (U_count - U_bin). U_bin = primer punto, U_count = último."""
    v_bin = curve[0]["value"]
    v_count = curve[-1]["value"]
    denom = v_count - v_bin
    # A flat curve (no counting benefit, e.g. B=1) has denom == 0 in exact
    # arithmetic, but different caps sum the DP over different bucketings, so
    # denom can be tiny FP noise instead of a clean 0. Treat anything within
    # tolerance as "no benefit to capture" (frac 0) rather than dividing by
    # noise, which would blow frac far outside [0, 1].
    out = []
    for pt in curve:
        frac = 0.0 if denom <= 1e-12 else (pt["value"] - v_bin) / denom
        out.append({"cap": pt["cap"], "value": pt["value"], "frac": frac})
    return out
