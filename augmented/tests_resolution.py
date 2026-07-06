#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests del núcleo D2 (curva de resolución).

Run:  /Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py
Exit code 0 = PASS, 1 = FAIL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import bin_of


def test_bin_of_truncation():
    # cap None => conteo completo (identidad)
    assert bin_of(0, None) == 0
    assert bin_of(3, None) == 3
    # cap = 1 => binario: 0 vs >=1
    assert bin_of(0, 1) == 0
    assert bin_of(1, 1) == 1
    assert bin_of(5, 1) == 1
    # cap = 2 => {0, 1, >=2}
    assert [bin_of(r, 2) for r in range(5)] == [0, 1, 2, 2, 2]
    # aísla {0}: ningún r>=1 cae en el bin 0
    for cap in (1, 2, 3):
        assert bin_of(0, cap) == 0
        assert all(bin_of(r, cap) >= 1 for r in range(1, 8))


from augmented.solver import solve_optimal_dapts
from augmented.classical_solver import solve_classical_dynamic


def _instance():
    # instancia chica, determinista (sin RNG): n=4, B=2, G=3
    p = [0.3, 0.5, 0.2, 0.7]
    u = [1.0, 2.0, 3.0, 1.0]
    return p, u, 2, 3


def test_cap_none_equals_counting():
    p, u, B, G = _instance()
    v_default, _ = solve_optimal_dapts(p, u, B, G)
    v_none, _ = solve_optimal_dapts(p, u, B, G, cap=None)
    assert abs(v_default - v_none) < 1e-12, (v_default, v_none)


def test_cap1_equals_classical_binary():
    p, u, B, G = _instance()
    v_cap1, _ = solve_optimal_dapts(p, u, B, G, cap=1)
    v_bin, _ = solve_classical_dynamic(p, u, B, G)
    assert abs(v_cap1 - v_bin) < 1e-9, (v_cap1, v_bin)


def test_cap_below_one_rejected():
    p, u, B, G = _instance()
    try:
        solve_optimal_dapts(p, u, B, G, cap=0)
    except ValueError:
        return
    raise AssertionError("cap=0 debió lanzar ValueError (fundiría {0,1})")


from augmented.simulator import apply_dapts


def test_truncated_policy_is_executable():
    # Una política con cap<G codifica sus historias por bins min(r,cap); el
    # replay canónico (apply_dapts) debe binear el conteo observado con el
    # mismo cuantizador. Instancia donde el pool jugado puede ver r>=2 bajo
    # cap=1 (antes: KeyError en la búsqueda de política).
    p = [0.4, 0.4, 0.4, 0.4, 0.4]
    u = [2.0, 2.0, 2.0, 2.0, 2.0]
    B, G = 2, 4
    value, F = solve_optimal_dapts(p, u, B, G, cap=1)
    assert F.cap == 1
    realized = 0.0
    q = [1.0 - pi for pi in p]
    for z in range(1 << len(p)):
        # no debe lanzar KeyError para ningún perfil realizable
        _, _, u_real = apply_dapts(F, z, len(p), u)
        wz = 1.0
        for i in range(len(p)):
            wz *= p[i] if (z >> i & 1) else q[i]
        realized += wz * u_real
    # el valor esperado replayado coincide con el valor del DP
    assert abs(realized - value) < 1e-9, (realized, value)


def test_flat_curve_fraction_stays_bounded():
    # Curva plana (sin beneficio del conteo): frac debe quedar en [0,1] aun si
    # denom es ruido de punto flotante en vez de 0 exacto.
    curve = [{"cap": 1, "value": 3.0}, {"cap": 2, "value": 3.0 + 1e-15},
             {"cap": 3, "value": 3.0}]
    fc = fraction_captured(curve)
    for pt in fc:
        assert 0.0 <= pt["frac"] <= 1.0, fc


from augmented.resolution_curve import cap_chain, resolution_curve, fraction_captured


def test_cap_chain():
    assert cap_chain(3) == [1, 2, 3]
    assert cap_chain(1) == [1]


def test_curve_monotone_nondecreasing():
    p, u, B, G = _instance()
    curve = resolution_curve(p, u, B, G)
    vals = [pt["value"] for pt in curve]
    # monotonía por refinamiento: U no decreciente en cap
    for a, b in zip(vals, vals[1:]):
        assert a <= b + 1e-12, vals
    assert [pt["cap"] for pt in curve] == cap_chain(G)


def test_fraction_endpoints():
    p, u, B, G = _instance()
    fc = fraction_captured(resolution_curve(p, u, B, G))
    assert abs(fc[0]["frac"] - 0.0) < 1e-9      # cap=1 (binario) captura 0
    assert abs(fc[-1]["frac"] - 1.0) < 1e-9     # conteo captura 1
    for pt in fc:
        assert -1e-9 <= pt["frac"] <= 1.0 + 1e-9


def test_b1_collapse():
    # §2: con B=1 el resultado no condiciona decisiones futuras; los tres
    # canales coinciden. La curva debe ser plana en B=1.
    p, u, G = [0.3, 0.5, 0.2, 0.7], [1.0, 2.0, 3.0, 1.0], 3
    curve = resolution_curve(p, u, 1, G)
    vals = [pt["value"] for pt in curve]
    assert max(vals) - min(vals) < 1e-9, vals


def test_curve_is_nontrivial_somewhere():
    # Con B>=2 debe existir una instancia donde el conteo separe al binario
    # (si no, no habría curva que medir). Instancia con horizonte y prior mixto.
    p, u, B, G = [0.05, 0.5, 0.5, 0.95], [3.0, 1.0, 1.0, 3.0], 2, 3
    curve = resolution_curve(p, u, B, G)
    assert curve[-1]["value"] - curve[0]["value"] > 1e-6, curve


from augmented.experiments_resolution import default_instances, sweep


def test_sweep_smoke():
    rows = sweep(default_instances()[:1])  # una instancia, rápido
    assert len(rows) >= 1
    keys = {"label", "n", "B", "G", "cap", "value", "frac"}
    assert keys.issubset(rows[0].keys()), rows[0]
    # respeta el régimen exacto (Global Constraints): n <= 6
    assert all(row["n"] <= 6 for row in rows)
    # chequeo barato (sin correr el barrido completo): TODAS las instancias
    # declaradas deben respetar el régimen exacto N<=6, no solo la primera.
    assert all(len(inst["p"]) <= 6 for inst in default_instances())


# ---- Test runner ----
if __name__ == "__main__":
    test_fns = sorted(
        [(name, obj) for name, obj in globals().items()
         if name.startswith("test_") and callable(obj)],
        key=lambda x: x[0],
    )
    passed = failed = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if failed:
        sys.exit(1)
