"""
Horizon experiment: ¿qué gobierna el beneficio del conteo (U_D_A - U_D)?

Aísla tres ejes — horizonte B, tamaño de pool G, población N — y un cuarto eje de
CONCENTRACIÓN de utilidad, midiendo el beneficio relativo (U_D_A-U_D)/U_D por
instancia (media, mediana, fracción de ceros). Hallazgo: el beneficio es un
fenómeno de HORIZONTE — cero exacto en B=1 y creciente con B, casi insensible a G
(satura pronto), y la concentración de utilidad lo COLAPSA.

Escribe results/horizon/horizon_sweep.csv (un renglón por punto de barrido).

Run:  PYTHONPATH=. python augmented/horizon_experiment.py [--instances K]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from augmented.solver import solve_optimal_dapts            # U_D_A (augmented)
from augmented.classical_solver import solve_classical_dynamic  # U_D (binario)


def benefit_stats(make_instance, ninst, seed):
    """make_instance(rng) -> (p, u, B, G). Devuelve (rel_mean, rel_median,
    frac_zero, abs_mean) del beneficio (U_D_A - U_D)."""
    rng = np.random.default_rng(seed)
    rels, abss, zeros = [], [], 0
    for _ in range(ninst):
        p, u, B, G = make_instance(rng)
        uda = solve_optimal_dapts(p, u, B, G)[0]
        ud = solve_classical_dynamic(p, u, B, G)[0]
        gap = uda - ud
        abss.append(gap)
        if ud > 1e-12:
            rels.append(gap / ud * 100.0)
        if gap < 1e-9:
            zeros += 1
    rels = np.array(rels) if rels else np.array([0.0])
    return float(rels.mean()), float(np.median(rels)), zeros / ninst, float(np.mean(abss))


def run(ninst=150, out_csv='results/horizon/horizon_sweep.csv'):
    rows = []

    def rand_pu(rng, n):
        return rng.uniform(0, 1, size=n).tolist(), rng.choice([1., 2., 3.], size=n).tolist()

    # Eje B (horizonte): n=6, G=4, B=1..3  (pools moderados para velocidad)
    for B in [1, 2, 3]:
        mk = lambda rng, B=B: (*rand_pu(rng, 6), B, 4)
        rm, md, fz, ab = benefit_stats(mk, ninst, seed=100 + B)
        rows.append({'axis': 'B', 'N': 6, 'B': B, 'G': 4,
                     'rel_mean': rm, 'rel_median': md, 'frac_zero': fz, 'abs_mean': ab})
        print(f"[B] B={B}: rel_mean={rm:.2f}% median={md:.2f}% frac0={fz:.2f}", flush=True)

    # Eje G (tamaño de pool): n=6, B=3, G=2..5
    for G in [2, 3, 4, 5]:
        mk = lambda rng, G=G: (*rand_pu(rng, 6), 3, G)
        rm, md, fz, ab = benefit_stats(mk, ninst, seed=300 + G)
        rows.append({'axis': 'G', 'N': 6, 'B': 3, 'G': G,
                     'rel_mean': rm, 'rel_median': md, 'frac_zero': fz, 'abs_mean': ab})
        print(f"[G] G={G}: rel_mean={rm:.2f}% median={md:.2f}% frac0={fz:.2f}", flush=True)

    # Eje N (poblacion): B=3, G=4, n=4..6
    for n in [4, 5, 6]:
        mk = lambda rng, n=n: (*rand_pu(rng, n), 3, 4)
        rm, md, fz, ab = benefit_stats(mk, ninst, seed=500 + n)
        rows.append({'axis': 'N', 'N': n, 'B': 3, 'G': 4,
                     'rel_mean': rm, 'rel_median': md, 'frac_zero': fz, 'abs_mean': ab})
        print(f"[N] n={n}: rel_mean={rm:.2f}% median={md:.2f}% frac0={fz:.2f}", flush=True)

    # Eje U (concentracion de utilidad): u=[u_top,1,...,1], n=6,B=3,G=4
    for utop in [1, 2, 3, 10, 100]:
        def mk(rng, utop=utop):
            p = rng.uniform(0, 1, size=6).tolist()
            u = [float(utop)] + [1.0] * 5
            return p, u, 3, 4
        rm, md, fz, ab = benefit_stats(mk, ninst, seed=900 + utop)
        rows.append({'axis': 'U', 'N': 6, 'B': 3, 'G': 4, 'u_top': utop,
                     'rel_mean': rm, 'rel_median': md, 'frac_zero': fz, 'abs_mean': ab})
        print(f"[U] u_top={utop}: rel_mean={rm:.2f}% frac0={fz:.2f} abs={ab:.4f}", flush=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ['axis', 'N', 'B', 'G', 'u_top', 'rel_mean', 'rel_median', 'frac_zero', 'abs_mean']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"\nwrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--instances', type=int, default=150)
    a = ap.parse_args()
    run(ninst=a.instances)
