# Augmented notebooks

Esta carpeta contiene los notebooks narrativos de `augmented/`. La idea es que
los notebooks sean el punto de entrada para entender los experimentos, mientras
que los modulos Python en `augmented/` siguen siendo la fuente del codigo.

## Orden recomendado

| Orden | Notebook | Proposito |
| --- | --- | --- |
| 01 | `01_examples_notebook.ipynb` | Introduccion a DAPTS, bitmasks, actualizacion Bayesiana y solvers. |
| 02 | `02_large_trees_exploration.ipynb` | Intuicion visual con arboles de decision pequenos y comparaciones greedy/optimo. |
| 03 | `04_combined_findings.ipynb` | Sintesis experimental amplia: efectos de valor, presupuesto, Gibbs, heuristicas y casos Nico. |
| 04 | `03_phase3_findings.ipynb` | Hallazgos tecnicos de Phase 3: Gibbs, beta, large G y gaps. |
| 05 | `05_heuristica_rl_combinado.ipynb` | Heuristica de independencia y consultas RL tabulares. |
| 06 | `08_vw.ipynb` | Idea VW de super-nodos, equivalencias y limitaciones. |
| 07 | `09_vw.ipynb` | Variante greedy/VW y primeras consultas de RL. |
| 08 | `06_paper_findings.ipynb` | Version curada para narrativa de paper. |
| 09 | `07_paper_findings_executed.ipynb` | Version ejecutada de `06_paper_findings.ipynb`, util para revisar outputs. |

## Bitacora cronologica (que se trabajo y cuando)

Orden real en que se creo cada notebook (por primer commit en git). El prefijo
`NN_` se dejo de usar tras `07_vw`; esta tabla recupera la numeracion para saber
la secuencia de trabajo. Cada notebook ya lleva su titulo `Notebook NN - ...` en
la primera celda, con el mismo NN cronologico de esta tabla.

| # | Fecha | Notebook | Titulo — que se trabajo |
| --- | --- | --- | --- |
| 01 | 2026-03-10 | `01_examples_notebook.ipynb` | Introduccion a DAPTS: bitmasks, actualizacion bayesiana y solvers exacto/greedy. |
| 02 | 2026-03-17 | `02_large_trees_exploration.ipynb` | Arboles de decision para DAPTS: intuicion visual y primeras comparaciones greedy vs optimo. |
| 03 | 2026-03-22 | `03_phase3_findings.ipynb` | Hallazgos tecnicos de Phase 3: Gibbs, parametro beta, G grande y brechas de optimalidad. |
| 04 | 2026-03-26 | `04_combined_findings.ipynb` | Sintesis experimental amplia: efecto del valor, presupuesto, Gibbs, heuristicas y casos de Nico. |
| 05 | 2026-05-04 | `05_heuristica_rl_combinado.ipynb` | Heuristica de independencia (producto de marginales) y consultas RL tabular. |
| 06 | 2026-05-04 | `06_paper_findings.ipynb` | Hallazgos curados para la narrativa del paper. |
| 07 | 2026-05-04 | `07_paper_findings_executed.ipynb` | Misma version ya ejecutada, para revisar outputs. |
| 08 | 2026-06-02 | `08_vw.ipynb` | Idea VW de super-nodos: equivalencias y limites. **JSON roto (linea 625): no parsea — pendiente reparar.** |
| 09 | 2026-06-02 | `09_vw.ipynb` | Variantes greedy/VW y primeras consultas RL. |
| 10 | 2026-06-08 | `10_nick_empirical_replication_augmented.ipynb` | Replicacion de los experimentos empiricos de Lopez en el modelo augmented (instancias chicas + algoritmos de conteo). |
| 11 | 2026-06-08 | `11_compendio.ipynb` | Compendio: un ejemplo clave de cada notebook, contado con su figura. |
| 12 | 2026-06-08 | `12_maestro.ipynb` | Cartas de discusion: ejemplos chicos con figura y preguntas abiertas para abrir direcciones. |
| 13 | 2026-06-08 | `13_resultados.ipynb` | Tres resultados del proyecto en graficas parametrizables (conteo vs si/no, presupuesto, ...). |
| 14 | 2026-06-22 | `14_separacion.ipynb` | La separacion estatica vs dinamica, derivada paso a paso. |
| 15 | 2026-06-23 | `15_descubrimiento.ipynb` | El valor de contar como fenomeno de horizonte (descubrimiento empirico). |
| 16 | 2026-06-24 | `16_intuicion_greedy.ipynb` | Que hace el greedy en instancias grandes: intuicion visual. |
| 17 | 2026-06-27 | `17_competencia.ipynb` | Los algoritmos compiten sobre una misma instancia, paso a paso. |
| 18 | 2026-07-07 | `18_sesion_francisco.ipynb` | El certificado computable (notebook de la sesion con Francisco). |
| 19 | 2026-07-09 | `19_avances_post_sesion.ipynb` | Avances tras la sesion del 9-jul: regimenes tratables, ruido, beta y certificado. |
| 20 | 2026-07-12 | `20_arboles_decision.ipynb` | Arboles verticales greedy vs optimo con la ruta realizada resaltada. |

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
