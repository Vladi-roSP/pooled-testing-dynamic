# Plan de implementación — Núcleo D2: la curva de resolución

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cuantizar el conteo del test en el DP exacto (familia de truncamiento min(r,k)) y producir la "curva de resolución" —utilidad óptima vs profundidad de resolución— en el régimen exacto, con controles de monotonía y colapso B=1.

**Architecture:** Un primitivo escalar `bin_of(r, cap)` (truncamiento) en `core.py`; un parámetro opcional `cap` en el DP `solve_optimal_dapts` que ramifica por bins en vez de por el conteo exacto y limpia cuando el bin observado es 0; un módulo `resolution_curve.py` que barre la cadena de caps y mide la fracción del beneficio capturada; un script de experimento que guarda un CSV y una figura. Sin cambios de fondo al DP: `cap=None` reproduce el conteo actual y `cap=1` reproduce el binario clásico.

**Tech Stack:** Python 3 (intérprete del repo: `/Users/hectorbecerrilvillamil/miniconda3/bin/python`), stdlib + matplotlib para la figura. Sin pytest: los tests son scripts standalone `tests_*.py` con un runner `__main__` que sale con código 1 si algo falla.

## Global Constraints

- **Todo corre en la M4.** Régimen exacto del DP hasta el límite práctico ~N = G = 5, B = 3 (N ≤ 6). No exceder eso en los resultados exactos.
- **El cuantizador debe AISLAR {0}.** La familia de truncamiento min(r,k) lo garantiza para todo cap ≥ 1. `cap = None` ⇒ conteo completo; `cap = 1` ⇒ binario. Un cap < 1 fundiría {0,1} y colapsaría la limpieza a 0: el solver debe **rechazarlo** con `ValueError`.
- **Intérprete:** `/Users/hectorbecerrilvillamil/miniconda3/bin/python`. Ejecutar tests desde la raíz del repo (`group-count-dynamic/`).
- **Estilo de imports:** `sys.path.insert(0, <raíz>)` y luego `from augmented.<mod> import ...`, igual que los `tests_*.py` existentes.
- **Compatibilidad hacia atrás:** `cap=None` por defecto; ninguna llamada existente a `solve_optimal_dapts` cambia de comportamiento.
- **Commits frecuentes:** un commit por tarea.

---

### Task 1: Primitivo `bin_of` (cuantizador de truncamiento) + arnés de tests

**Files:**
- Modify: `augmented/core.py` (añadir `bin_of` junto a `test_result`, ~L107)
- Create/Test: `augmented/tests_resolution.py`

**Interfaces:**
- Consumes: nada.
- Produces: `bin_of(r, cap)` → `int`. `cap is None` ⇒ devuelve `r` (conteo completo). `cap` entero ⇒ devuelve `min(r, cap)`. Con `cap ≥ 1`, `bin_of(0, cap) == 0` y `bin_of(r, cap) ≥ 1` para `r ≥ 1` (aísla {0}). `cap = 1` ⇒ binario (0 vs ≥1).

- [ ] **Step 1: Escribir el test que falla (crea el arnés `tests_resolution.py`)**

```python
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
```

- [ ] **Step 2: Correr el test para ver que falla**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: FAIL con `ImportError: cannot import name 'bin_of' from 'augmented.core'`.

- [ ] **Step 3: Implementar `bin_of` en `core.py`**

Justo debajo de `test_result` (~L109), añadir:

```python
def bin_of(r, cap):
    """Cuantizador de truncamiento del conteo r a un bin.

    cap is None  -> conteo completo (identidad): devuelve r.
    cap (int)    -> min(r, cap). Con cap >= 1 el bin 0 es exactamente {0}
                    (aísla {0}); cap = 1 es el régimen binario (0 vs >=1).
    """
    return r if cap is None else min(r, cap)
```

- [ ] **Step 4: Correr el test para ver que pasa**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: `PASS  test_bin_of_truncation` y `1 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add augmented/core.py augmented/tests_resolution.py
git commit -m "feat(resolution): bin_of truncation quantizer + test harness"
```

---

### Task 2: Parámetro `cap` en `solve_optimal_dapts` (DP cuantizado)

**Files:**
- Modify: `augmented/solver.py` (import de `bin_of`; firma; `dp()` y `reconstruct()`)
- Test: `augmented/tests_resolution.py` (añadir funciones `test_*`)

**Interfaces:**
- Consumes: `bin_of(r, cap)` (Task 1); `solve_classical_dynamic(p, u, B, G)` de `augmented/classical_solver.py` (existente, devuelve `(value, None)`; test binario 0/1).
- Produces: `solve_optimal_dapts(p, u, B, G, cap=None)` → `(optimal_value, policy)`. `cap=None` reproduce el conteo actual; `cap=1` reproduce el binario; `cap < 1` lanza `ValueError`. El historial en la política reconstruida usa el **bin** observado, no el conteo.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `augmented/tests_resolution.py` (antes del runner):

```python
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
```

- [ ] **Step 2: Correr los tests para ver que fallan**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: `test_cap_none_equals_counting` FAIL con `TypeError: solve_optimal_dapts() got an unexpected keyword argument 'cap'` (y los otros dos igual).

- [ ] **Step 3: Implementar `cap` en `solver.py`**

3a. Cambiar el import (L10):

```python
from augmented.core import all_pools, test_result, bin_of
```

3b. Cambiar la firma (L16) y añadir la validación tras el chequeo de `n`:

```python
def solve_optimal_dapts(p, u, B, G, cap=None):
    """Solve for the optimal DAPTS via brute-force DP.

    cap: cuantizador de truncamiento del conteo (min(r, cap)).
         None = conteo completo (augmented); 1 = binario clásico.
    Returns (optimal_value, optimal_policy).
    """
    n = len(p)
    if n > _MAX_N:
        raise ValueError(f"Brute-force requires n <= {_MAX_N}, got {n}")
    if cap is not None and cap < 1:
        raise ValueError(f"cap must isolate {{0}} (cap >= 1); got {cap}")
    if n == 0:
        return 0.0, DAPTS(B)
```

3c. En `dp()`, ramificar por bin (reemplazar el bloque L69-83):

```python
        for pool in pools:
            # Partition remaining profiles by observed bin
            buckets = {}
            for z in remaining:
                b = bin_of(test_result(pool, z), cap)
                buckets.setdefault(b, []).append(z)

            ev = 0.0
            for b, z_list in buckets.items():
                new_cleared = cleared_mask | pool if b == 0 else cleared_mask
                sub_val, _ = dp(k + 1, frozenset(z_list), new_cleared)
                ev += sub_val

            if ev > best_value:
                best_value, best_pool = ev, pool
```

3d. En `reconstruct()`, ramificar por bin (reemplazar L106-114):

```python
        buckets = {}
        for z in remaining:
            b = bin_of(test_result(best_pool, z), cap)
            buckets.setdefault(b, []).append(z)

        for b, z_list in buckets.items():
            new_cleared = cleared_mask | best_pool if b == 0 else cleared_mask
            reconstruct(k + 1, frozenset(z_list), new_cleared,
                        history + ((best_pool, b),))
```

(La limpieza usa `b == 0`, que ≡ `r == 0` porque todo `cap ≥ 1` —y `cap=None`— cumple `bin_of(0, cap) == 0` y `bin_of(r, cap) ≥ 1` para `r ≥ 1`.)

- [ ] **Step 4: Correr los tests para ver que pasan**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: `PASS` en `test_cap_none_equals_counting`, `test_cap1_equals_classical_binary`, `test_cap_below_one_rejected` (más el de Task 1). `4 passed, 0 failed`.

> Si `test_cap1_equals_classical_binary` falla, es una señal real: significa que `solve_classical_dynamic` y el DP con `cap=1` no ramifican/limpian igual (p.ej. distinto conjunto de pools o distinta opción de "desperdiciar test"). Reconciliar antes de seguir — `cap=1` DEBE reducirse al binario clásico.

- [ ] **Step 5: Commit**

```bash
git add augmented/solver.py augmented/tests_resolution.py
git commit -m "feat(resolution): quantized DP via cap param (cap=None=counting, cap=1=binary)"
```

---

### Task 3: Módulo `resolution_curve.py` (barrido de caps + fracción capturada)

**Files:**
- Create: `augmented/resolution_curve.py`
- Test: `augmented/tests_resolution.py` (añadir funciones `test_*`)

**Interfaces:**
- Consumes: `solve_optimal_dapts(p, u, B, G, cap)` (Task 2).
- Produces:
  - `cap_chain(G)` → `list[int]` = `[1, 2, ..., G]` (caps con sentido: un pool tiene a lo más G miembros, así que `cap ≥ G` ≡ conteo completo).
  - `resolution_curve(p, u, B, G, caps=None)` → `list[dict]` con claves `{'cap': int, 'value': float}`, una entrada por cap (por defecto `cap_chain(G)`). El cap máximo se resuelve con `cap=None` (conteo) y se etiqueta con `cap=G`.
  - `fraction_captured(curve)` → `list[dict]` con `{'cap', 'value', 'frac'}`, donde `frac = (value - v_bin) / (v_count - v_bin)` usando el primer punto (cap=1, binario) y el último (conteo). Si `v_count == v_bin`, `frac = 0.0` para todos.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `augmented/tests_resolution.py`:

```python
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
```

- [ ] **Step 2: Correr los tests para ver que fallan**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'augmented.resolution_curve'`.

- [ ] **Step 3: Implementar `resolution_curve.py`**

```python
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
    out = []
    for pt in curve:
        frac = 0.0 if denom <= 0 else (pt["value"] - v_bin) / denom
        out.append({"cap": pt["cap"], "value": pt["value"], "frac": frac})
    return out
```

- [ ] **Step 4: Correr los tests para ver que pasan**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: `7 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add augmented/resolution_curve.py augmented/tests_resolution.py
git commit -m "feat(resolution): resolution_curve + fraction_captured over cap chain"
```

---

### Task 4: Controles — colapso B=1 y curva no trivial

**Files:**
- Test: `augmented/tests_resolution.py` (añadir funciones `test_*`)

**Interfaces:**
- Consumes: `solve_optimal_dapts` (Task 2), `resolution_curve` (Task 3).
- Produces: dos tests de caracterización que blindan corolarios del spec (§2 colapso B=1; §5 la curva mide algo real).

- [ ] **Step 1: Escribir los tests de control**

Añadir a `augmented/tests_resolution.py`:

```python
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
```

- [ ] **Step 2: Correr los tests**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: ambos PASS. `9 passed, 0 failed`.

> `test_b1_collapse` debe pasar por construcción del modelo. Si falla, hay un bug en la limpieza/ramificación del DP cuantizado (Task 2), no en el test. `test_curve_is_nontrivial_somewhere` verifica que el conteo separa en al menos un régimen; si falla, consulta otra instancia con más horizonte (B=3) antes de dudar del modelo, y anótalo.

- [ ] **Step 3: Commit**

```bash
git add augmented/tests_resolution.py
git commit -m "test(resolution): B=1 collapse + non-trivial-curve characterization controls"
```

---

### Task 5: Experimento — barrido y CSV de la curva

**Files:**
- Create: `augmented/experiments_resolution.py`
- Create (salida): `augmented/data/resolution_curve.csv`
- Test: `augmented/tests_resolution.py` (smoke test de la función de barrido)

**Interfaces:**
- Consumes: `resolution_curve`, `fraction_captured` (Task 3).
- Produces:
  - `sweep(instances)` → `list[dict]` con `{'label', 'n', 'B', 'G', 'cap', 'value', 'frac'}` (una fila por (instancia, cap)).
  - `default_instances()` → `list[dict]` con instancias deterministas (sin RNG) en el régimen exacto: N ≤ 6, B ∈ {1,2,3}, G ≤ 5, con etiquetas de prevalencia (baja/media/alta).
  - `main()` que corre `sweep(default_instances())`, imprime la tabla y escribe `augmented/data/resolution_curve.csv`.

- [ ] **Step 1: Escribir el smoke test que falla**

Añadir a `augmented/tests_resolution.py`:

```python
from augmented.experiments_resolution import default_instances, sweep


def test_sweep_smoke():
    rows = sweep(default_instances()[:1])  # una instancia, rápido
    assert len(rows) >= 1
    keys = {"label", "n", "B", "G", "cap", "value", "frac"}
    assert keys.issubset(rows[0].keys()), rows[0]
    # respeta el régimen exacto (Global Constraints): n <= 6
    assert all(row["n"] <= 6 for row in rows)
```

- [ ] **Step 2: Correr el test para ver que falla**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'augmented.experiments_resolution'`.

- [ ] **Step 3: Implementar `experiments_resolution.py`**

```python
"""Barrido de la curva de resolución en el régimen exacto (N<=6). Escribe un
CSV con U_k y la fracción del beneficio del conteo capturada por cada cap."""

import csv
import os

from augmented.resolution_curve import resolution_curve, fraction_captured

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def default_instances():
    """Instancias deterministas en el régimen exacto (N<=6, B en {1,2,3})."""
    return [
        {"label": "baja_n4", "p": [0.05, 0.1, 0.08, 0.12], "u": [1, 2, 3, 1], "B": 2, "G": 3},
        {"label": "media_n4", "p": [0.3, 0.5, 0.2, 0.7], "u": [1, 2, 3, 1], "B": 2, "G": 3},
        {"label": "media_n5", "p": [0.3, 0.5, 0.2, 0.7, 0.4], "u": [1, 2, 3, 1, 2], "B": 3, "G": 3},
        {"label": "alta_n5", "p": [0.6, 0.7, 0.5, 0.8, 0.55], "u": [1, 2, 3, 1, 2], "B": 3, "G": 3},
        {"label": "horizonte_b1_n4", "p": [0.3, 0.5, 0.2, 0.7], "u": [1, 2, 3, 1], "B": 1, "G": 3},
    ]


def sweep(instances):
    rows = []
    for inst in instances:
        p, u, B, G = inst["p"], inst["u"], inst["B"], inst["G"]
        fc = fraction_captured(resolution_curve(p, u, B, G))
        for pt in fc:
            rows.append({
                "label": inst["label"], "n": len(p), "B": B, "G": G,
                "cap": pt["cap"], "value": round(pt["value"], 6),
                "frac": round(pt["frac"], 6),
            })
    return rows


def main():
    rows = sweep(default_instances())
    os.makedirs(_DATA_DIR, exist_ok=True)
    out = os.path.join(_DATA_DIR, "resolution_curve.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "n", "B", "G", "cap", "value", "frac"])
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)
    print(f"\nEscrito {out} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el smoke test (pasa) y el experimento completo**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py`
Expected: `10 passed, 0 failed`.

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/experiments_resolution.py`
Expected: imprime las filas y `Escrito .../data/resolution_curve.csv`. Verificar a ojo: para `horizonte_b1_n4` todos los `frac` son 0 (colapso B=1); para las demás, `frac` crece de 0 (cap=1) a 1 (cap=G) de forma no decreciente.

- [ ] **Step 5: Commit**

```bash
git add augmented/experiments_resolution.py augmented/tests_resolution.py augmented/data/resolution_curve.csv
git commit -m "feat(resolution): sweep experiment + resolution_curve.csv"
```

---

### Task 6: Figura de la curva de resolución

**Files:**
- Create: `augmented/figures_resolution.py`
- Create (salida): `augmented/figures/resolution_curve.png`

**Interfaces:**
- Consumes: `default_instances`, `sweep` (Task 5).
- Produces: `plot_resolution_curve(path)` que dibuja `frac` vs `cap` (una línea por instancia con B≥2) y guarda un PNG.

- [ ] **Step 1: Implementar la figura**

```python
"""Figura de la curva de resolución: fracción del beneficio del conteo vs
profundidad de resolución (cap)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augmented.experiments_resolution import default_instances, sweep

_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def plot_resolution_curve(path=None):
    rows = sweep(default_instances())
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, pts in by_label.items():
        if pts[0]["B"] < 2:
            continue  # B=1 es plano (colapso); no aporta a la figura
        pts = sorted(pts, key=lambda r: r["cap"])
        ax.plot([p["cap"] for p in pts], [p["frac"] for p in pts],
                marker="o", label=label)
    ax.set_xlabel("profundidad de resolución (cap = k en min(r, k))")
    ax.set_ylabel("fracción del beneficio del conteo")
    ax.set_title("Curva de resolución (régimen exacto)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if path is None:
        os.makedirs(_FIG_DIR, exist_ok=True)
        path = os.path.join(_FIG_DIR, "resolution_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Escrito", plot_resolution_curve())
```

- [ ] **Step 2: Generar la figura y verificar**

Run: `/Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/figures_resolution.py`
Expected: imprime `Escrito .../figures/resolution_curve.png`.

Run: `test -s augmented/figures/resolution_curve.png && echo OK`
Expected: `OK` (el PNG existe y no está vacío).

- [ ] **Step 3: Commit**

```bash
git add augmented/figures_resolution.py augmented/figures/resolution_curve.png
git commit -m "feat(resolution): resolution curve figure"
```

---

## Planes de seguimiento (fuera de este plan)

Este plan entrega el **núcleo D2** (curva de resolución exacta) completo y testeable. Los otros tres subsistemas del spec van en planes separados, cada uno reutilizando `bin_of` y el parámetro `cap`/`pools`:

1. **D1 — perilla K:** parámetro `pools=None` en `solve_optimal_dapts` + generador de pools de grado ≤ K; colapso K=1 y array K=2 (2×2/3×3). (§6 del spec.)
2. **D3 — benchmarks honestos:** `solve_hindsight(p, u, B, G)` (cota U_PI) + reproducción del contraejemplo de no-adaptive-submodularity. (§7.)
3. **§8 — TV + mixing:** diagnóstico de convergencia del Gibbs basado en TV (upgrade de `gibbs_analysis.py`) + (stretch) exploración del mixing time ligada a K. (§8.)

---

## Self-review (cobertura del spec, D2)

- **§3 modelo (cuantizador por umbral, aísla {0}):** Task 1 (`bin_of` truncamiento) + Task 2 (validación `cap ≥ 1`). ✓
- **§4 lema (monotonía por refinamiento):** Task 3 `test_curve_monotone_nondecreasing` (control numérico del lema). ✓
- **§5 experimento (curva, escala exacta, cadena min(r,k), métricas):** Tasks 3–5 (`resolution_curve`, `fraction_captured`, barrido CSV, `cap=None`=conteo, `cap=1`=binario). ✓
- **§2 colapso B=1 (corolario):** Task 4 `test_b1_collapse`. ✓
- **§2/§5 el conteo separa (curva no trivial):** Task 4 `test_curve_is_nontrivial_somewhere`. ✓
- **Restricción "cap=1 = binario clásico":** Task 2 `test_cap1_equals_classical_binary` (contra `solve_classical_dynamic`). ✓
- **Figura:** Task 6. ✓
- Fuera de alcance (planes de seguimiento): U_PI/hindsight, solver-K, contraejemplo D3, diagnóstico TV/mixing — correctamente diferidos.
