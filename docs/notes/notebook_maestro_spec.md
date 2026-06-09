# Spec — Notebook maestro (sesión con Marmolejo)

_2026-06-03. Diseño aprobado (enfoque C: recap navegable + temas profundos). En
español, lenguaje simple (estilo `estado_proyecto_2026-05.md`). Híbrido:
ejemplos N≤5 ejecutables en vivo; tablas/figuras pesadas embebidas o cacheadas.
Cada implementación lleva un marcador **📍 Estado / dónde lo dejamos**._

## Objetivo

Un solo notebook que recapitule el proyecto de **Dynamic Augmented Pooled
Testing** para una sesión con Francisco Marmolejo, cubriendo 6 pedidos:

1. Ejemplos de **cómo sabemos que Gibbs funciona**.
2. **Perfiles de infección con mayor separación** (3 nociones).
3. **Separación promedio**.
4. **Recapitulación** de las implementaciones importantes de todos los notebooks.
5. **En qué punto dejamos cada cosa**.
6. **1 ejemplo pequeño significativo** de cada implementación, estilo tesis
   (N≤5, diagramas de pools, aritmética explícita, árboles de decisión).

## Decisiones de alcance (confirmadas con el usuario)

- **Separación** → mostrar las **tres** nociones como secciones distintas
  (4.1 distinguibilidad de perfiles, 4.2 brecha entre estrategias, 4.3
  independence gap), dejando claro que son cosas diferentes.
- **Ejecución** → **híbrido**: ejemplos chicos en vivo + lo pesado embebido.
- **VW** → una sección más del recap (sin énfasis especial), pero con las
  preguntas abiertas resaltadas en el estado final.

## Estructura (`augmented/notebooks/notebook_maestro.ipynb`)

- **0. Portada + propósito + mini-glosario + setup** — `sys.path` robusto a la
  raíz del repo, imports verificados, semilla fija, instancia canónica chica.
- **1. El modelo en 1 minuto** — augmented = el test da el **conteo exacto**;
  objetivo de bienestar; cadena `U_single ≤ U_static ≤ U_greedy ≤ U_óptimo`.
  Ejemplo estilo Ejemplo 1.1 de la tesis (N=3, B=2, G=3).
- **2. Recap con 1 ejemplo chico de cada implementación** (pedidos #4 + #6):
  - 2.1 Solver óptimo DP (`solve_optimal_dapts`) + árbol de decisión.
  - 2.2 Greedy dinámico: miope / por conteo / gibbs.
  - 2.3 Bayes: exacto (info cruzada) vs Gibbs.
  - 2.4 VW super-nodos (Francisco): VW-A ≡ greedy miope.
  - 2.5 RL pedagógico: value iteration (=DP) y Q-learning (se acerca).
  - 2.6 Baselines/estáticos: la cadena de desigualdad.
- **3. ¿Cómo sabemos que Gibbs funciona?** (pedido #1) — Gibbs vs Bayes exacto
  (curva de convergencia ejecutable), bug de mixing + fallback exacto, 64/64
  tests, n=8 ~500 iters, ~78× más rápido.
- **4. Separación (3 nociones; c/u con "mayor separación" + "promedio")**
  (pedidos #2, #3) — 4.1 distinguibilidad de perfiles, 4.2 brecha entre
  estrategias, 4.3 independence gap.
- **5. Estado: qué quedó a medias / abierto** (pedido #5) — tabla de
  `estado_proyecto §3`, con las 3 dudas abiertas del VW + garantía (1−1/e)
  resaltadas para Marmolejo.
- **6. Apéndice: índice de los notebooks originales.**

## Fuentes / API verificada (imports `from augmented.X import Y`, desde la raíz)

- `solver.solve_optimal_dapts(p,u,B,G)`
- `greedy.greedy_myopic_expected_utility` / `_counting_` / `_gibbs_` `(p,u,B,G)`
- `bayesian.bayesian_update(p,history,n)` (exacto, info cruzada),
  `bayesian_update_by_counting`, `gibbs_update(p,history,n,num_iterations,...)`
- `independence_gap.exact_pool_pmf` / `independence_pool_pmf`
- `baselines.u_max` / `u_single`; `static_solver.solve_static_non_overlapping` /
  `_overlapping`; `classical_solver.solve_classical_dynamic`
- `vw_restrict.restriction_experiment`; `vw_restrict_sweep.REGIMES` / `run_regime`
- `rl_examples` (value iteration / Q-learning); `tree_extractor` / `tree_visualizer`
- `core.mask_from_indices` / `mask_str` / `all_pools` / `compute_active_mask`

## Garantía de calidad

- Todo número y ejemplo proviene de **código ejecutado** (no memorizado).
- Cada celda en vivo corre en <~20s; lo más lento se precomputa y se embebe.
- El notebook se ejecuta de punta a punta (`nbconvert --execute`) y debe correr
  limpio antes de darse por terminado.

## Convención de construcción

Seguir el patrón del repo: script `build_maestro_notebook.py` con `nbformat`
(como `build_05_notebook.py` / `build_vw_notebook.py`), luego ejecutar el
notebook para embeber salidas.
