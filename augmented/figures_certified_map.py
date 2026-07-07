"""Figura del mapa con garantias: fraccion real vs certificada por (B, cap).

Un panel por presupuesto B. En cada panel, la curva de resolucion real
(OPT(cap)/OPT(G), gris punteada) y la certificada (OPT(cap)/U_pen, azul
solida); la banda entre ambas es el hueco de demostracion que la linea de
certificados busca cerrar. Lee data/certified_map.csv (generado por
experiments_certified_map.py).
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_HERE, "data", "certified_map.csv")
_FIG_DIR = os.path.join(_HERE, "figures")

_AZUL = "#2563eb"
_GRIS = "#6b7280"


def plot_certified_map(path=None):
    with open(_CSV) as f:
        rows = [{k: float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]
    budgets = sorted({int(r["B"]) for r in rows})
    caps = sorted({int(r["cap"]) for r in rows})

    fig, axes = plt.subplots(1, len(budgets), figsize=(9, 3.2),
                             sharey=True)
    for ax, B in zip(axes, budgets):
        real, cert = [], []
        for cap in caps:
            sub = [r for r in rows if r["B"] == B and r["cap"] == cap]
            real.append(sum(r["real_frac"] for r in sub) / len(sub))
            cert.append(sum(r["cert_frac"] for r in sub) / len(sub))
        ax.fill_between(caps, cert, real, color=_AZUL, alpha=0.10)
        ax.plot(caps, real, marker="s", ms=4, ls="--", color=_GRIS,
                label="real  OPT(cap)/OPT")
        ax.plot(caps, cert, marker="o", ms=4, color=_AZUL,
                label="certificada  OPT(cap)/U_pen")
        ax.set_title(f"B = {B}", fontsize=10)
        ax.set_xlabel("resolución (cap)")
        ax.set_xticks(caps)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("fracción del valor")
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle("El mapa con garantías: valor real vs valor certificable "
                 f"(n={5}, G={3}, media de 12 instancias)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if path is None:
        os.makedirs(_FIG_DIR, exist_ok=True)
        path = os.path.join(_FIG_DIR, "certified_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Escrito", plot_certified_map())
