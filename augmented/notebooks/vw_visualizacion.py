#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualizacion del ejemplo VW (super-nodos): agentes como puntos, pools como
elipses, y DOS iteraciones del scoring VW-A con formacion de super-nodo.

VW-A (all-clear) score de un pool t:  (prod_{i in t} (1 - q_i)) * (sum_{i in t} u_i)
  = prob. de que el pool salga LIMPIO por la utilidad que se declara limpia de golpe,
  usando el posterior actual q_i (no el prior, una vez hay historia).

Tras testear, los agentes tocados forman un super-nodo S con posterior CONJUNTO.
La instancia (posiciones + membresias) esta arriba y es editable.
"""
import sys
ROOT = "/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic"
sys.path.insert(0, ROOT)

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from augmented.bayesian import bayesian_update_by_counting

# ----------------------------------------------------------------------
# INSTANCIA (lectura del dibujo; editar si hace falta)
# ----------------------------------------------------------------------
POS = {
    0: (3.2, 1.0), 1: (3.0, 5.7), 2: (5.4, 8.1), 3: (6.7, 5.2), 4: (3.2, 3.4),
    5: (4.8, 6.6), 6: (5.7, 3.6), 7: (7.4, 7.2), 8: (1.9, 4.2),
    9: (6.6, 1.2), 10: (1.4, 6.2), 11: (1.7, 8.4), 12: (3.3, 8.5),
}
POOLS = {"t1": [2, 5], "t2": [1, 4], "t3": [3, 6], "t4": [0, 4, 6, 9]}
N = len(POS)
BUDGET = 2

rng = np.random.default_rng(7)
p = {i: round(float(rng.uniform(0.15, 0.55)), 2) for i in POS}   # prior P(estado activo)
u = {i: int(rng.integers(1, 4)) for i in POS}                    # utilidad {1,2,3}
Z = {i: int(rng.random() < p[i]) for i in POS}                   # estado real (oculto)

POOL_COLORS = {"t1": "#1d6fb8", "t2": "#2a9d8f", "t3": "#e76f51", "t4": "#8e44ad"}


def mask(ids):
    m = 0
    for i in ids:
        m |= (1 << i)
    return m


def ellipse_params(ids, pad=1.2):
    pts = np.array([POS[i] for i in ids], float)
    c = pts.mean(0)
    if len(pts) == 1:
        return c, 1.0, 1.0, 0.0
    d = pts - c
    vals, vecs = np.linalg.eigh(np.cov(d.T) + 1e-6 * np.eye(2))
    proj = d @ vecs
    h = 2 * np.abs(proj[:, 0]).max() + pad
    w = 2 * np.abs(proj[:, 1]).max() + pad
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return c, w, h, ang


def draw_pool(ax, ids, color, lw=2.2, alpha=0.12, ls="-"):
    c, w, h, ang = ellipse_params(ids)
    ax.add_patch(Ellipse(c, w, h, angle=ang, facecolor=color, alpha=alpha,
                         edgecolor=color, lw=lw, ls=ls, zorder=1))
    return c


def draw_supernode(ax, ids):
    c, w, h, ang = ellipse_params(ids, pad=1.9)
    ax.add_patch(Ellipse(c, w, h, angle=ang, facecolor="none",
                         edgecolor="k", lw=2.4, ls=(0, (4, 3)), zorder=2))
    ax.text(c[0], c[1] - h / 2 - 0.25, "super-nodo S", ha="center",
            fontsize=11, fontstyle="italic", zorder=5)


def draw_agents(ax, post, cleared):
    for i, (x, y) in POS.items():
        if i in cleared:
            col, ec = "#2ecc71", "#1e8449"
        else:
            col, ec = plt.cm.RdYlGn_r(post[i]), "k"
        ax.scatter([x], [y], s=270, c=[col], edgecolors=ec, linewidths=1.2, zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, zorder=4)


def vw_score(ids, q):
    clean = float(np.prod([1 - q[i] for i in ids]))
    util = sum(u[i] for i in ids)
    return clean * util, clean, util


def supernode_joint(S, history):
    """Posterior CONJUNTO sobre el super-nodo S (enumeracion exacta)."""
    S = sorted(S)
    rows = []
    for bits in itertools.product([0, 1], repeat=len(S)):
        st = dict(zip(S, bits))
        ok = all(sum(st[j] for j in POOLS_by_mask[pm] if j in st) == r
                 for pm, r in history)
        if not ok:
            continue
        w = 1.0
        for j, b in st.items():
            w *= p[j] if b else (1 - p[j])
        rows.append((bits, w))
    tot = sum(w for _, w in rows)
    return S, [(b, w / tot) for b, w in rows]


POOLS_by_mask = {mask(ids): ids for ids in POOLS.values()}


def figure(fname, title, q, cleared, chosen, tested_before, supernode):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, ids in POOLS.items():
        faded = name in tested_before
        c = draw_pool(ax, ids, POOL_COLORS[name],
                      lw=1.3 if faded else 2.2, alpha=0.06 if faded else 0.12)
        lbl = name
        if name == chosen:
            lbl = name + "  <- elegido"
        elif faded:
            lbl = name + " (ya testeado)"
        ax.text(c[0], c[1] + 0.1, lbl, color=POOL_COLORS[name],
                fontsize=12, fontweight="bold", ha="center", zorder=5)
    if chosen:
        draw_pool(ax, POOLS[chosen], "k", lw=2.6, alpha=0.0)
    if supernode:
        draw_supernode(ax, supernode)
    draw_agents(ax, q, cleared)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="posterior P(estado activo)")
    ax.set_title(title)
    ax.set_xlim(0.3, 8.4); ax.set_ylim(0, 9.3); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def run():
    print("Estado real Z (oculto):", {i: Z[i] for i in sorted(Z) if Z[i]} or "todos limpios",
          "(activos)")
    history, tested, S = [], set(), set()

    for it in range(1, BUDGET + 1):
        q_vec = bayesian_update_by_counting([p[i] for i in range(N)], tuple(history), N)
        q = {i: q_vec[i] for i in range(N)}
        avail = {n: ids for n, ids in POOLS.items() if n not in tested}
        print(f"\n===== ITERACION {it} =====")
        if it > 1:
            print("Posterior tras la historia previa (marginales que cambiaron):")
            for i in sorted(S):
                if abs(q[i] - p[i]) > 1e-9:
                    print(f"   agente {i}: prior {p[i]:.2f} -> posterior {q[i]:.3f}")
        print("Score VW-A por pool disponible (prod(1-q) * sum u):")
        scores = {}
        for n, ids in avail.items():
            s, clean, util = vw_score(ids, q)
            scores[n] = s
            tag = "  [toca S]" if (S & set(ids)) else ""
            print(f"   {n} {ids}:  P(limpio)={clean:.3f} * U={util} = {s:.3f}{tag}")
        best = max(scores, key=scores.get)
        ids = POOLS[best]
        r = sum(Z[i] for i in ids)
        print(f"-> elige {best} = {ids} (score {scores[best]:.3f}); conteo observado r = {r}")

        tested_before = set(tested)
        S_before = set(S)
        history.append((mask(ids), r))
        tested.add(best)
        S |= set(ids)

        q2_vec = bayesian_update_by_counting([p[i] for i in range(N)], tuple(history), N)
        q2 = {i: q2_vec[i] for i in range(N)}
        cleared = set(i for pm, rr in history for i in POOLS_by_mask[pm]) if False else set()
        cleared = set(i for i in range(N) if q2[i] < 1e-9)
        figure(f"vw_iter{it}.png",
               f"Iteracion {it}: VW-A elige {best}, r={r}",
               q2, cleared, best, tested_before, S_before if S_before else None)

    # Super-nodo final y su posterior CONJUNTO
    print("\n===== SUPER-NODO FINAL =====")
    Ss, joint = supernode_joint(S, history)
    print(f"S = {Ss}  (agentes tocados por la historia)")
    print("Posterior conjunto sobre S (perfiles consistentes con los conteos):")
    for bits, prob in sorted(joint, key=lambda x: -x[1]):
        infset = [Ss[k] for k in range(len(Ss)) if bits[k]]
        print(f"   activos={infset or '{}'}:  P = {prob:.3f}")
    shared = [i for i in Ss if sum(i in v for v in POOLS.values()) > 1]
    print(f"Agentes de S compartidos entre pools: {shared} (donde VW debe usar la CONJUNTA, no el producto).")
    print("\nFiguras: vw_iter_setup.png, vw_iter1.png, vw_iter2.png")


def setup_figure():
    q = {i: p[i] for i in range(N)}
    figure("vw_iter_setup.png", "Ejemplo VW: agentes (puntos) y pools (elipses)",
           q, set(), None, set(), None)


if __name__ == "__main__":
    setup_figure()
    run()
