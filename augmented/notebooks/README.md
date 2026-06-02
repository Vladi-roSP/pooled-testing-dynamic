# Augmented notebooks

Esta carpeta contiene los notebooks narrativos de `augmented/`. La idea es que
los notebooks sean el punto de entrada para entender los experimentos, mientras
que los modulos Python en `augmented/` siguen siendo la fuente del codigo.

## Orden recomendado

| Orden | Notebook | Proposito |
| --- | --- | --- |
| 01 | `examples_notebook.ipynb` | Introduccion a DAPTS, bitmasks, actualizacion Bayesiana y solvers. |
| 02 | `large_trees_exploration.ipynb` | Intuicion visual con arboles de decision pequenos y comparaciones greedy/optimo. |
| 03 | `combined_findings.ipynb` | Sintesis experimental amplia: efectos de valor, presupuesto, Gibbs, heuristicas y casos Nico. |
| 04 | `phase3_findings.ipynb` | Hallazgos tecnicos de Phase 3: Gibbs, beta, large G y gaps. |
| 05 | `05_heuristica_rl_combinado.ipynb` | Heuristica de independencia y pruebas RL tabulares. |
| 06 | `06_vw.ipynb` | Idea VW de super-nodos, equivalencias y limitaciones. |
| 07 | `07_vw.ipynb` | Variante greedy/VW y primeras pruebas de RL. |
| 08 | `paper_findings.ipynb` | Version curada para narrativa de paper. |
| 09 | `paper_findings_executed.ipynb` | Version ejecutada de `paper_findings.ipynb`, util para revisar outputs. |

## Formato comun

Todos los notebooks principales deben empezar con:

1. Titulo claro.
2. Bloque breve de **objetivo**, **pregunta guia** y **lectura esperada**.
3. Celda `Setup` con imports, path del repo, `matplotlib` y parametros globales.
4. Secciones numeradas con una explicacion corta antes de cada bloque de codigo.
5. Comentarios de codigo directos, al estilo de la parte `dynamic`: variables
   matematicas (`n`, `B`, `G`, `p`, `u`), ejemplos pequenos primero y resultados
   impresos cerca del calculo.

## Artefactos

- `paper_figs/` guarda figuras curadas para el paper.
- Los PNG `combined_*`, `phase3_*` y `07_vw_fig*` son salidas de notebooks.
- Los scripts `build_*.py` regeneran notebooks especificos.
- Los `.tex` y auxiliares LaTeX (`.aux`, `.out`, `.log`) se quitaron de esta
  carpeta para dejar GitHub enfocado en notebooks y figuras.
