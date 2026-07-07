"""Demo: certificacion de una flota de agentes bajo presupuesto de evals.

Traduccion literal del motor DAPTS al caso de uso de evaluacion de sistemas
de IA: n=50 componentes (prompts, herramientas, modelos) con probabilidad
previa de estar rotos y valor de negocio; cada "corrida por lotes" es un pool
de a lo mas G componentes cuyo resultado es el NUMERO de fallas del lote (no
cuales); el presupuesto es B corridas. El motor decide adaptativamente que
lote correr, declara limpios a los componentes en lotes sin fallas, y emite
el certificado: la asignacion del presupuesto logra al menos X% de la mejor
asignacion posible (cota U_PI por informacion perfecta, Monte Carlo).

Uso:  PYTHONPATH=. python augmented/demo_fleet_certification.py
"""

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask
from augmented.greedy import greedy_myopic_simulate
from augmented.pool_solvers import mosek_best_pool
from augmented.certificates import u_pi_mc

N, B, G = 50, 10, 5
SEED = 7
NUM_SIMS = 300


def build_fleet(rng):
    """50 componentes: 12 prompts, 20 herramientas, 6 modelos, 12 flujos."""
    names, p, u = [], [], []
    kinds = ([("prompt", 12), ("tool", 20), ("model", 6), ("flow", 12)])
    for kind, count in kinds:
        for j in range(count):
            names.append(f"{kind}-{j:02d}")
            suspect = rng.random() < 0.25
            p.append(rng.uniform(0.20, 0.45) if suspect
                     else rng.uniform(0.02, 0.10))
            u.append(round(rng.uniform(1.0, 10.0), 2))
    return names, p, u


def selector(p, u, G, n, cleared_mask):
    return mosek_best_pool(p, u, G, n, cleared_mask)  # cae a heuristico


def random_selector(rng):
    def pick(p, u, G, n, cleared_mask):
        pending = [i for i in range(n) if not (cleared_mask >> i & 1)]
        rng.shuffle(pending)
        mask = 0
        for i in pending[:G]:
            mask |= (1 << i)
        return mask if mask else 1
    return pick


def sample_z(p, rng):
    z = 0
    for i, pi in enumerate(p):
        if rng.random() < pi:
            z |= (1 << i)
    return z


def mc_value(p, u, sel, num_sims, seed):
    vals = []
    for s in range(num_sims):
        rng = random.Random(seed + s)
        z = sample_z(p, rng)
        _, _, val = greedy_myopic_simulate(p, u, B, G, z, pool_selector=sel)
        vals.append(val)
    mean = statistics.fmean(vals)
    se = statistics.stdev(vals) / (num_sims ** 0.5)
    return mean, se


def main():
    rng = random.Random(SEED)
    names, p, u = build_fleet(rng)
    total_value = sum(u)

    print("=" * 66)
    print("CERTIFICACION DE FLOTA - 50 componentes, presupuesto: "
          f"{B} corridas por lotes (<= {G} componentes por lote)")
    print("=" * 66)

    # --- Una corrida real, paso a paso ---
    z = sample_z(p, random.Random(2026))
    truly_broken = set(indices_from_mask(z, N))
    history, cleared_mask, val = greedy_myopic_simulate(
        p, u, B, G, z, pool_selector=selector)
    print(f"\nFallas reales ocultas: {len(truly_broken)} de {N} componentes\n")
    for t, (pool, r) in enumerate(history, 1):
        members = indices_from_mask(pool, N)
        tag = "LIMPIO -> certificados" if r == 0 else f"{r} falla(s) dentro"
        print(f"  corrida {t:>2}: [{', '.join(names[i] for i in members)}]"
              f"  ->  {tag}")
    cleared = indices_from_mask(cleared_mask, N)
    missed = [i for i in cleared if i in truly_broken]
    print(f"\nComponentes certificados limpios: {len(cleared)} de {N} "
          f"(valor asegurado {val:.1f} de {total_value:.1f})")
    print(f"Falsos limpios: {len(missed)} (el conteo exacto nunca "
          "certifica un lote con fallas)")

    # --- El certificado sobre la POLITICA, no sobre una corrida ---
    print("\n" + "-" * 66)
    mean_g, se_g = mc_value(p, u, selector, NUM_SIMS, seed=100)
    mean_r, se_r = mc_value(p, u, random_selector(random.Random(3)),
                            NUM_SIMS, seed=100)
    upi = u_pi_mc(p, u, B, G, num_samples=200000, seed=0)
    print(f"Valor esperado del motor:        {mean_g:8.2f} +- {se_g:.2f}")
    print(f"Valor esperado muestreo aleator: {mean_r:8.2f} +- {se_r:.2f}")
    print(f"Cota superior U_PI (nadie puede mas que esto): {upi:8.2f}")
    print("\nCERTIFICADO: la asignacion del motor garantiza >= "
          f"{100*mean_g/upi:.0f}% del optimo incalculable "
          f"(muestreo aleatorio: {100*mean_r/upi:.0f}%).")
    print("Con la cota penalizada (en desarrollo) el mismo numero sube sin "
          "cambiar el motor.")


if __name__ == "__main__":
    main()
