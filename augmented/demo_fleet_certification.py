"""Demo: certificacion de una flota de agentes bajo presupuesto de evals.

Traduccion literal del motor DAPTS al caso de uso de evaluacion de sistemas
de IA: n=50 componentes (prompts, herramientas, modelos) con probabilidad
previa de estar rotos y valor de negocio; cada "corrida por lotes" es un pool
de a lo mas G componentes cuyo resultado es el NUMERO de fallas del lote (no
cuales); el presupuesto es B corridas. El motor decide adaptativamente que
lote correr, declara limpios a los componentes en lotes sin fallas, y emite
el certificado: la asignacion del presupuesto logra al menos X% de la mejor
asignacion posible (cota U_PI por informacion perfecta).

Cableado (2026-07): acarreo de creencias exacto por componentes
(gibbs_update), seleccion de pool por frecuencia conjunta sobre draws del
posterior (posterior_draws + sample_best_pool) y certificado contra U_PI en
forma cerrada (regimen saturado) o Monte Carlo con error estandar (escasez).
Sin dependencias de entorno silenciosas: la configuracion se imprime en la
cabecera.

Uso:  PYTHONPATH=. python augmented/demo_fleet_certification.py
"""

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import indices_from_mask
from augmented.greedy import greedy_myopic_simulate
from augmented.bayesian import gibbs_update, posterior_draws
from augmented.pool_solvers import sample_best_pool
import augmented.pool_solvers as pool_solvers
from augmented.certificates import u_pi_mc, _pi_welfare

N = 50
SEED = 7
NUM_SIMS = 300          # ~0.1 s por sim con el selector conjunto (medido)
SELECTOR_DRAWS = 1000
UPI_MC_SAMPLES = 200000

# (B, G, etiqueta): el regimen saturado (cap >= n) se conserva por
# continuidad; el de escasez (cap < n) es el titular — ahi U_PI no colapsa a
# U_max y la cota penalizada muerde (hallazgo de holgura, 2026-07-07).
REGIMES = [
    (10, 5, "saturado  (cap 50 >= n=50)"),
    (6, 5, "escasez   (cap 30 <  n=50)"),
]


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


class JointSelector:
    """Selector consciente de la historia: draws del posterior exacto +
    frecuencia conjunta muestral. La historia le llega por su callback
    belief_update (greedy_myopic_simulate no se la pasa al selector), asi que
    cada simulacion necesita su propia instancia."""

    def __init__(self, prior, seed, num_draws=SELECTOR_DRAWS):
        self.prior = list(prior)
        self.seed = seed
        self.num_draws = num_draws
        self.history = ()

    def belief_update(self, prior, history, n):
        self.history = history
        return gibbs_update(prior, history, n, seed=0)

    def __call__(self, cur_p, u, G, n, cleared_mask):
        step_seed = self.seed + 1000 * len(self.history)
        draws = posterior_draws(self.prior, self.history, n,
                                num_draws=self.num_draws, seed=step_seed)
        return sample_best_pool(draws, u, G, n, cleared_mask)


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


def mc_value(p, u, B, G, make_sel, num_sims, seed):
    """Media +- SE del valor de la politica; make_sel(s) entrega el par
    (pool_selector, belief_update) fresco de la simulacion s."""
    vals = []
    for s in range(num_sims):
        rng = random.Random(seed + s)
        z = sample_z(p, rng)
        pool_selector, belief_update = make_sel(s)
        _, _, val = greedy_myopic_simulate(p, u, B, G, z,
                                           pool_selector=pool_selector,
                                           belief_update=belief_update)
        vals.append(val)
    mean = statistics.fmean(vals)
    se = statistics.stdev(vals) / (num_sims ** 0.5)
    return mean, se


def u_pi_mc_se(p, u, B, G, num_samples, seed):
    """u_pi_mc con error estandar (acumulador de segundo momento). En regimen
    saturado delega en la forma cerrada exacta (SE = 0)."""
    n = len(p)
    cap = B * G
    if cap >= n:
        return u_pi_mc(p, u, B, G), 0.0
    rng = random.Random(seed)
    acc = 0.0
    acc2 = 0.0
    for _ in range(num_samples):
        w = _pi_welfare(sample_z(p, rng), u, n, cap)
        acc += w
        acc2 += w * w
    mean = acc / num_samples
    var = max(0.0, acc2 / num_samples - mean * mean)
    return mean, (var / num_samples) ** 0.5


def certificate_block(p, u, B, G, label):
    print("\n" + "-" * 66)
    print(f"Regimen {label}: B={B} corridas, lotes <= {G}")
    mean_g, se_g = mc_value(
        p, u, B, G,
        lambda s: (lambda sel: (sel, sel.belief_update))(
            JointSelector(p, seed=10_000 * B + s)),
        NUM_SIMS, seed=100)
    mean_r, se_r = mc_value(
        p, u, B, G,
        lambda s: (random_selector(random.Random(3 + s)), None),
        NUM_SIMS, seed=100)
    upi, upi_se = u_pi_mc_se(p, u, B, G, UPI_MC_SAMPLES, seed=0)
    upi_tag = ("exacta (forma cerrada)" if upi_se == 0.0
               else f"Monte Carlo +- {upi_se:.2f}")
    print(f"Valor esperado del motor:        {mean_g:8.2f} +- {se_g:.2f}")
    print(f"Valor esperado muestreo aleator: {mean_r:8.2f} +- {se_r:.2f}")
    print(f"Cota superior U_PI ({upi_tag}): {upi:8.2f}")
    print(f"CERTIFICADO: la asignacion del motor garantiza >= "
          f"{100*mean_g/upi:.0f}% del optimo incalculable "
          f"(muestreo aleatorio: {100*mean_r/upi:.0f}%).")
    return mean_g, upi


def main():
    rng = random.Random(SEED)
    names, p, u = build_fleet(rng)
    total_value = sum(u)

    print("=" * 66)
    print("CERTIFICACION DE FLOTA - 50 componentes")
    print("=" * 66)
    print("Configuracion de inferencia (sin dependencias silenciosas):")
    print(f"  seleccion de pool : posterior_draws(S={SELECTOR_DRAWS}) + "
          "sample_best_pool (frecuencia conjunta)")
    print("  acarreo de creencias: gibbs_update (deducciones + componentes "
          "exactos <= 16 agentes)")
    print(f"  backend pool_solvers: "
          f"{pool_solvers.LAST_BACKEND or 'no usado (selector muestral)'}")
    print(f"  simulaciones MC por politica: {NUM_SIMS}")

    # --- Una corrida real, paso a paso (regimen saturado, continuidad) ---
    B, G = REGIMES[0][0], REGIMES[0][1]
    z = sample_z(p, random.Random(2026))
    truly_broken = set(indices_from_mask(z, N))
    showcase = JointSelector(p, seed=2026)
    history, cleared_mask, val = greedy_myopic_simulate(
        p, u, B, G, z, pool_selector=showcase,
        belief_update=showcase.belief_update)
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
    for B, G, label in REGIMES:
        certificate_block(p, u, B, G, label)
    print("\nCon la cota penalizada (en desarrollo) el numero de escasez "
          "sube sin cambiar el motor.")


if __name__ == "__main__":
    main()
