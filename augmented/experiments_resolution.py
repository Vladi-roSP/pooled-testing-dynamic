#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Barrido de la curva de resolución en el régimen exacto (N<=6). Escribe un
CSV con U_k y la fracción del beneficio del conteo capturada por cada cap."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # G=4: cap_chain=[1,2,3,4] tiene dos niveles interiores (cap=2,3);
        # a G=3 el único interior (cap=2) siempre satura en frac=1.0.
        {"label": "media_n5_g4", "p": [0.3, 0.5, 0.4, 0.6, 0.35], "u": [1, 2, 3, 1, 2], "B": 3, "G": 4},
        {"label": "alta_n5_g4", "p": [0.5, 0.6, 0.55, 0.65, 0.45], "u": [3, 1, 2, 1, 2], "B": 3, "G": 4},
        # media_n5_g4 y alta_n5_g4 (n=5) saturan igual en cap=2 (frac=1.0):
        # a n=5 con G=4 el pool máximo (4 de 5 activos) deja muy poco margen
        # para que r=3 sea informativo. Con n=6 sí aparece un punto interior
        # genuino (cap=2 < 1.0 antes de saturar en cap=3).
        {"label": "mixta_n6_g4", "p": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65], "u": [1, 2, 3, 1, 2, 3], "B": 3, "G": 4},
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
