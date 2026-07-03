"""Figura de la curva de resolución: fracción del beneficio del conteo vs
profundidad de resolución (cap)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
