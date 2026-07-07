"""El mapa con garantias: fraccion real vs certificada sobre (B, cap).

Para una familia de instancias (n=5, G=3) computa, por presupuesto B y
resolucion cap:

  real(B, cap)  = OPT(cap) / OPT(G)      — la curva de resolucion (D2),
                                            solo computable en n chico
  cert(B, cap)  = OPT(cap) / U_pen(B)    — lo que el certificado garantiza
                                            (D3), computable a cualquier escala

Como U_pen >= OPT(G), vale cert <= real fila por fila; el hueco entre ambas
curvas es exactamente lo que falta por demostrar. Escribe
data/certified_map.csv.

Uso:  PYTHONPATH=. python augmented/experiments_certified_map.py
"""

import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.solver import solve_optimal_dapts
from augmented.certificates import u_pi_exact, u_pen_exact

N, G = 5, 3
BUDGETS = (1, 2, 3)
NUM_INSTANCES = 12
BASE_SEED = 42
SCALES = (0.25, 0.5, 1.0, 2.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "certified_map.csv")


def instances():
    # Prevalencia moderada-alta: el regimen donde el conteo paga (la
    # separacion aumentado-clasico enciende en rho~0.15 y crece hasta ~0.40),
    # asi que la curva de resolucion tiene estructura visible.
    out = []
    for i in range(NUM_INSTANCES):
        rng = random.Random(BASE_SEED + i)
        p = [rng.uniform(0.25, 0.65) for _ in range(N)]
        u = [rng.uniform(1.0, 5.0) for _ in range(N)]
        out.append((i, p, u))
    return out


def main():
    rows = []
    for B in BUDGETS:
        t0 = time.time()
        for i, p, u in instances():
            upi = u_pi_exact(p, u, B, G)
            upen = u_pen_exact(p, u, B, G, v_hat="umax", scales=SCALES)
            opt_full, _ = solve_optimal_dapts(p, u, B, G)
            assert upen >= opt_full - 1e-9, f"U_pen invalida (B={B}, i={i})"
            for cap in range(1, G + 1):
                opt_cap, _ = solve_optimal_dapts(p, u, B, G, cap=cap)
                real = opt_cap / opt_full if opt_full > 0 else 1.0
                cert = opt_cap / upen if upen > 0 else 1.0
                assert cert <= real + 1e-9, f"cert>real (B={B}, i={i}, cap={cap})"
                rows.append({
                    "B": B, "cap": cap, "instance": i,
                    "opt_cap": opt_cap, "opt_full": opt_full,
                    "u_pi": upi, "u_pen": upen,
                    "real_frac": real, "cert_frac": cert,
                })
        print(f"B={B}: {NUM_INSTANCES} instancias en {time.time()-t0:.0f}s",
              file=sys.stderr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {OUT}  ({len(rows)} filas)")

    print(f"\n{'':>6}" + "".join(f"cap={c:<10}" for c in range(1, G + 1)))
    for B in BUDGETS:
        line = f"B={B}   "
        for cap in range(1, G + 1):
            sub = [r for r in rows if r["B"] == B and r["cap"] == cap]
            real = sum(r["real_frac"] for r in sub) / len(sub)
            cert = sum(r["cert_frac"] for r in sub) / len(sub)
            line += f"{real:.2f}/{cert:.2f}    "
        print(line + "   (real/certificada)")


if __name__ == "__main__":
    main()
