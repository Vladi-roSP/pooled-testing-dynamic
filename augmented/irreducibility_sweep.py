"""Barrido de irreducibilidad del generador de caminos alternantes.

Para cada topologia (conjunto de soportes de measurements) y cada fibra
(clase de perfiles con la misma firma de conteos), este script:

  1. Enumera TODOS los movimientos que el generador real puede emitir con
     probabilidad positiva desde cada estado (replica fiel de
     _propose_alternating_move: pending con orden de insercion de dict,
     "primer measurement desbalanceado", elegibles, max_steps), ramificando
     sobre cada eleccion aleatoria.
  2. Verifica VALIDEZ: todo movimiento emitido aterriza dentro de la fibra.
  3. Verifica LEMA 2: todo movimiento kernel indescomponible aplicable
     (delta = z' - z sin sub-vector propio balanceado) es emitido.
  4. Verifica SIMETRIA: si delta es emitible desde z, -delta lo es desde z+delta
     (el camino espejo es una corrida valida).
  5. Verifica CONECTIVIDAD: BFS dirigido sobre los movimientos emitibles
     alcanza toda la fibra.

Un fallo en 3 refuta el Lema 2 (la reduccion); un fallo en 5 es una fibra
disconexa (contraejemplo de irreducibilidad). Cero fallos = evidencia
exhaustiva para el teorema.
"""
import itertools
import random
import sys
import time
from collections import deque

POP = (int.bit_count if hasattr(int, "bit_count")
       else (lambda x: bin(x).count("1")))


# ----------------------------------------------------------------------
# Replica fiel del generador (rama por rama, con backtracking)
# ----------------------------------------------------------------------

def emittable_moves(n, members, agent_tests, z, max_steps, node_cap=2_000_000):
    """Conjunto de (plus_mask, minus_mask) emitibles con prob > 0 desde z.

    Fidelidad con _propose_alternating_move (bayesian.py:385):
      - a0 recorre todos los agentes (rng.choice(comp) -> rama por cada a0)
      - delta: agente -> {-1,+1}, cada agente se voltea a lo mas una vez
      - pending: dict measurement -> desbalance, ORDEN DE INSERCION preservado
        (claves creadas al tocar el measurement; copia/undo respeta el orden)
      - ti = primer measurement desbalanceado en orden de insercion (415-419)
      - want = -1 si imbalance > 0, si no +1 (421)
      - elegibles: miembros fuera de delta con el valor de estado correcto (423-426)
      - sin elegibles -> rama muerta (427-428); pending todo cero -> emite (435)
      - presupuesto de pasos max_steps (417-418)
    """
    out = set()
    nodes = 0
    capped = False

    delta = {}
    pending = {}

    def dfs(steps):
        nonlocal nodes, capped
        nodes += 1
        if nodes > node_cap:
            capped = True
            return
        ti = next((t for t, v in pending.items() if v != 0), None)
        if ti is None:
            plus = minus = 0
            for a, d in delta.items():
                if d > 0:
                    plus |= 1 << a
                else:
                    minus |= 1 << a
            out.add((plus, minus))
            return
        if steps + 1 > max_steps:
            return
        imb = pending[ti]
        want = -1 if imb > 0 else 1
        need_bit = 1 if want == -1 else 0
        for a in members[ti]:
            if a in delta or ((z >> a) & 1) != need_bit:
                continue
            delta[a] = want
            added = []
            for tj in agent_tests[a]:
                if tj in pending:
                    pending[tj] += want
                else:
                    pending[tj] = want
                    added.append(tj)
            dfs(steps + 1)
            for tj in agent_tests[a]:
                if tj not in added:
                    pending[tj] -= want
            for tj in added:
                del pending[tj]
            del delta[a]

    for a0 in range(n):
        d0 = 1 - 2 * ((z >> a0) & 1)
        delta.clear()
        pending.clear()
        delta[a0] = d0
        for ti in agent_tests[a0]:
            pending[ti] = pending.get(ti, 0) + d0
        dfs(0)
    return out, capped


# ----------------------------------------------------------------------
# Indescomponibilidad de un movimiento
# ----------------------------------------------------------------------

def is_indecomposable(plus, minus, supports):
    """True si delta (codificado plus/minus) NO tiene sub-vector propio no
    vacio balanceado en todos los measurements."""
    S = plus | minus
    sub = (S - 1) & S
    while sub:
        if all(POP(plus & s & sub) == POP(minus & s & sub) for s in supports):
            return False
        sub = (sub - 1) & S
    return True


# ----------------------------------------------------------------------
# Analisis de una topologia completa (todas sus fibras a la vez)
# ----------------------------------------------------------------------

def incidence_connected(n, supports):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in supports:
        agents = [i for i in range(n) if s >> i & 1]
        for a in agents[1:]:
            ra, rb = find(agents[0]), find(a)
            if ra != rb:
                parent[rb] = ra
    return len({find(i) for i in range(n)}) == 1


def analyze_topology(n, supports, lemma2_pair_cap, rng, stats):
    m = len(supports)
    members = [[i for i in range(n) if s >> i & 1] for s in supports]
    agent_tests = [[t for t in range(m) if supports[t] >> a & 1]
                   for a in range(n)]
    max_steps = 6 * n + 12

    fibers = {}
    for z in range(1 << n):
        sig = tuple(POP(z & s) for s in supports)
        fibers.setdefault(sig, []).append(z)

    for sig, states in fibers.items():
        if len(states) < 2:
            continue
        stats["fibers"] += 1
        if len({POP(z) for z in states}) > 1:
            stats["multilevel_fibers"] += 1
        states_set = set(states)
        emit = {}
        for z in states:
            mv, capped = emittable_moves(n, members, agent_tests, z, max_steps)
            if capped:
                stats["caps"] += 1
            emit[z] = mv
            for (plus, minus) in mv:
                z2 = (z & ~minus) | plus
                if z2 not in states_set:
                    stats["validity_violations"].append(
                        (n, supports, sig, z, (plus, minus)))
        # simetria (camino espejo)
        for z in states:
            for (plus, minus) in emit[z]:
                z2 = (z & ~minus) | plus
                if (minus, plus) not in emit[z2]:
                    stats["mirror_violations"].append(
                        (n, supports, sig, z, (plus, minus)))
        # lema 2: todo indescomponible aplicable es emitible
        pairs = [(za, zb) for za in states for zb in states if za != zb]
        if len(pairs) > lemma2_pair_cap:
            pairs = rng.sample(pairs, lemma2_pair_cap)
            stats["lemma2_sampled_fibers"] += 1
        for za, zb in pairs:
            plus = zb & ~za
            minus = za & ~zb
            if is_indecomposable(plus, minus, supports):
                stats["lemma2_checks"] += 1
                if (plus, minus) not in emit[za]:
                    stats["lemma2_violations"].append(
                        (n, supports, sig, za, zb))
        # conectividad dirigida
        seen = {states[0]}
        dq = deque([states[0]])
        while dq:
            z = dq.popleft()
            for (plus, minus) in emit[z]:
                z2 = (z & ~minus) | plus
                if z2 not in seen:
                    seen.add(z2)
                    dq.append(z2)
        if len(seen) != len(states):
            stats["disconnected_fibers"].append((n, supports, sig))


def new_stats():
    return {"topologies": 0, "fibers": 0, "multilevel_fibers": 0,
            "lemma2_checks": 0, "lemma2_sampled_fibers": 0, "caps": 0,
            "validity_violations": [], "mirror_violations": [],
            "lemma2_violations": [], "disconnected_fibers": []}


def report(label, stats, t0):
    ok = (not stats["validity_violations"] and not stats["mirror_violations"]
          and not stats["lemma2_violations"]
          and not stats["disconnected_fibers"])
    print(f"[{label}] topologias={stats['topologies']} "
          f"fibras>=2={stats['fibers']} (multinivel={stats['multilevel_fibers']}) "
          f"lema2_checks={stats['lemma2_checks']} "
          f"{'TODO OK' if ok else 'VIOLACIONES!'} "
          f"({time.time() - t0:.1f}s)")
    for key in ("validity_violations", "mirror_violations",
                "lemma2_violations", "disconnected_fibers"):
        for v in stats[key][:5]:
            print(f"    {key}: {v}")
    if stats["caps"]:
        print(f"    ATENCION: {stats['caps']} estados con DFS truncado (cap)")
    return ok


# ----------------------------------------------------------------------
# Generadores de topologias
# ----------------------------------------------------------------------

def exhaustive(n, ms, lemma2_pair_cap=100000):
    rng = random.Random(0)
    stats = new_stats()
    t0 = time.time()
    supports_all = [s for s in range(1, 1 << n) if POP(s) >= 2]
    full = (1 << n) - 1
    for m in ms:
        for combo in itertools.combinations(supports_all, m):
            u = 0
            for s in combo:
                u |= s
            if u != full or not incidence_connected(n, combo):
                continue
            stats["topologies"] += 1
            analyze_topology(n, combo, lemma2_pair_cap, rng, stats)
    return report(f"n={n} m={list(ms)} EXHAUSTIVO", stats, t0)


def sampled(n, ms, k_per_m, seed, lemma2_pair_cap=400):
    rng = random.Random(seed)
    stats = new_stats()
    t0 = time.time()
    supports_all = [s for s in range(1, 1 << n) if POP(s) >= 2]
    full = (1 << n) - 1
    for m in ms:
        got = 0
        tries = 0
        seen_combos = set()
        while got < k_per_m and tries < 60 * k_per_m:
            tries += 1
            combo = tuple(sorted(rng.sample(supports_all, m)))
            if combo in seen_combos:
                continue
            u = 0
            for s in combo:
                u |= s
            if u != full or not incidence_connected(n, combo):
                continue
            seen_combos.add(combo)
            got += 1
            stats["topologies"] += 1
            analyze_topology(n, combo, lemma2_pair_cap, rng, stats)
    return report(f"n={n} m={list(ms)} MUESTREADO k={k_per_m}", stats, t0)


def adversarial_families():
    """Cadenas impares/pares (registro adversarial) hasta n=8: m = n-1
    measurements {i,i+1}=*, fuera del rango m<=4 del barrido general."""
    rng = random.Random(1)
    stats = new_stats()
    t0 = time.time()
    for n in range(3, 9):
        supports = tuple((0b11 << i) for i in range(n - 1))
        stats["topologies"] += 1
        analyze_topology(n, supports, 100000, rng, stats)
    # triples solapados del registro (n=5) ya cae en el barrido general
    return report("cadenas n=3..8 (familia adversarial)", stats, t0)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "A"
    all_ok = True
    if phase == "A":
        all_ok &= exhaustive(3, (2, 3, 4))
        all_ok &= exhaustive(4, (2, 3, 4))
        all_ok &= adversarial_families()
    elif phase == "B":
        all_ok &= exhaustive(5, (2, 3, 4))
    elif phase == "C":
        all_ok &= exhaustive(6, (2,))
        all_ok &= sampled(6, (3, 4), 2500, seed=11, lemma2_pair_cap=2000)
        all_ok &= sampled(7, (2, 3, 4), 1200, seed=22, lemma2_pair_cap=600)
        all_ok &= sampled(8, (2, 3, 4), 700, seed=33, lemma2_pair_cap=300)
    print("RESULTADO GLOBAL:", "OK" if all_ok else "FALLO")
