#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construye notebook_maestro.ipynb (REEMPLAZA el anterior) como CARTAS DE
DISCUSION: cada carta = figura + minimo texto + preguntas abiertas. Lee las
celdas verificadas del workflow (discusion_cells.json: lista de
{id,title,setup_md,code,questions_md}).
"""
import html
import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "maestro_cells.json")
OUT = os.path.join(HERE, "notebook_maestro.ipynb")


def _u(s):
    return html.unescape(s or "")


PORTADA = r"""# Cartas de discusion — Dynamic Augmented Pooled Testing

Cada carta es un ejemplo chico con su figura y un par de preguntas abiertas,
pensado para discutir y abrir direcciones. Poco texto: la figura manda.
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
"""


def main():
    with open(CELLS, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cards = data["cards"] if isinstance(data, dict) and "cards" in data else data

    cells = [nbf.v4.new_markdown_cell(PORTADA), nbf.v4.new_code_cell(SETUP)]
    for c in cards:
        head = _u(c.get("setup_md")) or _u(c.get("title"))
        if _u(c.get("title")) and _u(c.get("title")) not in head:
            head = _u(c.get("title")) + "\n\n" + head
        cells.append(nbf.v4.new_markdown_cell(head.strip()))
        src = _u(c.get("code")).strip()
        if src:
            cells.append(nbf.v4.new_code_cell(src))
        # Las "Preguntas para explorar" NO van en el notebook; viven en
        # preguntas_personales.md (fuera de git).

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python"}
    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Escrito {OUT} con {len(cells)} celdas ({len(cards)} cartas).")


if __name__ == "__main__":
    main()
