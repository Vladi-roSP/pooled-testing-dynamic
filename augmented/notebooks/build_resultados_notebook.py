#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construye notebook_resultados.ipynb (graficas primero) a partir de las celdas
verificadas por el workflow (resultados_cells.json: lista de {id,title,intro_md,
code,takeaway_md}).
"""
import html
import json
import os

import nbformat as nbf


def _u(s):
    return html.unescape(s or "")

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "resultados_cells.json")
OUT = os.path.join(HERE, "notebook_resultados.ipynb")

PORTADA = r"""# Resultados (graficas) — Dynamic Augmented Pooled Testing

Tres resultados del proyecto, cada uno como una grafica. Cada celda tiene sus
parametros al inicio: cambialos y vuelve a correr para explorar. Todo sale de
correr el codigo del paquete `augmented`.
"""

SETUP = r"""%matplotlib inline
import os, sys
_d = os.path.abspath(os.getcwd())
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "augmented", "__init__.py")):
    _d = os.path.dirname(_d)
if not os.path.isfile(os.path.join(_d, "augmented", "__init__.py")):
    _fb = "/Users/hectorbecerrilvillamil/Desktop/PooledTesting/pooled-testing-dynamic"
    if os.path.isfile(os.path.join(_fb, "augmented", "__init__.py")):
        _d = _fb
if _d not in sys.path:
    sys.path.insert(0, _d)
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0); np.random.seed(0)
print("repo root:", _d)
"""


def main():
    with open(CELLS, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cells_data = data["cells"] if isinstance(data, dict) and "cells" in data else data

    cells = [
        nbf.v4.new_markdown_cell(PORTADA),
        nbf.v4.new_code_cell(SETUP),
    ]
    for c in cells_data:
        intro = (_u(c.get("title", "")) + "\n\n" + _u(c.get("intro_md"))).strip()
        cells.append(nbf.v4.new_markdown_cell(intro))
        src = _u(c.get("code")).strip()
        if src:
            cells.append(nbf.v4.new_code_cell(src))
        tk = _u(c.get("takeaway_md")).strip()
        if tk:
            cells.append(nbf.v4.new_markdown_cell(tk))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python"}
    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Escrito {OUT} con {len(cells)} celdas ({len(cells_data)} graficas).")


if __name__ == "__main__":
    main()
