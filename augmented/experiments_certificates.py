"""Barrido de certificados en el regimen exacto: OPT vs greedy vs U_PI vs U_pen.

Para cada configuracion (n, B, G) genera instancias aleatorias y computa las
cuatro cantidades. Reporta la fraccion certificada por hindsight
(greedy/U_PI) y por la cota penalizada (greedy/U_pen), junto con la fraccion
real (greedy/OPT) que solo es computable en n chico. Escribe
data/certificates_small_n.csv y imprime la tabla resumen.

Uso:  PYTHONPATH=. python augmented/experiments_certificates.py
"""

import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.solver import solve_optimal_dapts
from augmented.greedy import greedy_myopic_expected_utility
from augmented.certificates import u_pi_exact, u_pen_exact

CONFIGS = [
    # (n, B, G, num_instances)
    (4, 2, 2, 20),
    (4, 2, 3, 20),
    (5, 2, 3, 20),
    (5, 3, 3, 20),
    (6, 2, 3, 20),
    (6, 3, 3, 6),    # caro: pocas instancias
]
SCALES = (0.25, 0.5, 1.0, 2.0)
BASE_SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "certificates_small_n.csv")


def random_instance(n, rng):
    p = [rng.uniform(0.05, 0.60) for _ in range(n)]
    u = [rng.uniform(1.0, 5.0) for _ in range(n)]
    return p, u


def main():
    rows = []
    for (n, B, G, k) in CONFIGS:
        t0 = time.time()
        for i in range(k):
            rng = random.Random(BASE_SEED + 1000 * n + 100 * B + 10 * G + i)
            p, u = random_instance(n, rng)
            opt, _ = solve_optimal_dapts(p, u, B, G)
            grd = greedy_myopic_expected_utility(p, u, B, G)
            upi = u_pi_exact(p, u, B, G)
            upen = u_pen_exact(p, u, B, G, v_hat="umax", scales=SCALES)
            assert upi >= opt - 1e-9, f"U_PI invalida en n={n},B={B},G={G},i={i}"
            assert upen >= opt - 1e-9, f"U_pen invalida en n={n},B={B},G={G},i={i}"
            rows.append({
                "n": n, "B": B, "G": G, "instance": i,
                "opt": opt, "greedy": grd, "u_pi": upi, "u_pen": upen,
                "true_ratio": grd / opt if opt > 0 else 1.0,
                "cert_pi": grd / upi if upi > 0 else 1.0,
                "cert_pen": grd / upen if upen > 0 else 1.0,
            })
        print(f"config n={n} B={B} G={G}: {k} instancias en "
              f"{time.time()-t0:.0f}s", file=sys.stderr)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {OUT}  ({len(rows)} filas)\n")

    print(f"{'config':<14}{'greedy/OPT':>12}{'cert U_PI':>12}"
          f"{'cert U_pen':>12}{'apriete':>10}")
    for (n, B, G, k) in CONFIGS:
        sub = [r for r in rows if r["n"] == n and r["B"] == B and r["G"] == G]
        mean = lambda key: sum(r[key] for r in sub) / len(sub)
        tr, cpi, cpen = mean("true_ratio"), mean("cert_pi"), mean("cert_pen")
        print(f"n={n} B={B} G={G} {tr:>12.3f}{cpi:>12.3f}{cpen:>12.3f}"
              f"{(cpen-cpi):>+10.3f}")


if __name__ == "__main__":
    main()
