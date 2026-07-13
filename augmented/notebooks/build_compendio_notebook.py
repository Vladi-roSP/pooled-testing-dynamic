#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Construye notebook_compendio.ipynb: un ejemplo clave por notebook previo,
grafica primero, texto breve. Lee compendio_cells.json ({cards:[{id,title,
intro_md,code}]}).
"""
import html
import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "compendio_cells.json")
OUT = os.path.join(HERE, "notebook_compendio.ipynb")


def _u(s):
    return html.unescape(s or "")


PORTADA = r"""# Notebook 11 - Compendio: un ejemplo clave de cada notebook, con su figura

Un ejemplo clave de cada notebook del proyecto, contado con su figura y un par de
frases. La grafica manda; el texto solo la enmarca. Todo sale de correr el codigo
del paquete `augmented`.
"""

SETUP = r"""%matplotlib inline
import os, sys
_d = os.path.abspath(os.getcwd())
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "augmented", "__init__.py")):
    _d = os.path.dirname(_d)
if not os.path.isfile(os.path.join(_d, "augmented", "__init__.py")):
    _fb = "/Users/hectorbecerrilvillamil/Desktop/GroupCounting/group-count-dynamic"
    if os.path.isfile(os.path.join(_fb, "augmented", "__init__.py")):
        _d = _fb
if _d not in sys.path:
    sys.path.insert(0, _d)
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0); np.random.seed(0)
"""

# Orden de presentacion (de lo introductorio a lo avanzado)
ORDER = [
    "examples_notebook", "nick_empirical_replication_augmented", "large_trees_exploration",
    "05_heuristica_rl_combinado", "06_vw", "07_vw", "combined_findings", "phase3_findings",
]


def main():
    with open(CELLS, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cards = data["cards"] if isinstance(data, dict) and "cards" in data else data
    by_id = {c.get("id"): c for c in cards}
    ordered = [by_id[i] for i in ORDER if i in by_id]
    ordered += [c for c in cards if c.get("id") not in ORDER]  # cualquier extra

    cells = [nbf.v4.new_markdown_cell(PORTADA), nbf.v4.new_code_cell(SETUP)]
    for c in ordered:
        intro = (_u(c.get("title", "")) + "\n\n" + _u(c.get("intro_md"))).strip()
        # evita duplicar el encabezado si intro_md ya empieza con '## '
        if _u(c.get("intro_md")).strip().startswith("##"):
            intro = _u(c.get("intro_md")).strip()
        cells.append(nbf.v4.new_markdown_cell(intro))
        src = _u(c.get("code")).strip()
        if src:
            cells.append(nbf.v4.new_code_cell(src))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python"}
    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Escrito {OUT} con {len(cells)} celdas ({len(ordered)} cartas).")


if __name__ == "__main__":
    main()
