# Recopilación de algoritmos: más allá del greedy miope

El greedy miope (`_myopic_best_pool`, que maximiza P(r=0)·Σu) es el tronco de todo el árbol: cada variante es una rama que cambia un solo eje —anticipación, motor de creencias, recompensa, super-nodos, óptimo de referencia o aprendizaje— sin tocar necesariamente los demás.

## 1. Variantes de greedy en `augmented/greedy.py`

El núcleo es `_myopic_best_pool` y todas las demás funciones lo invocan como selector por defecto (`pool_selector=None`). Tres ejes ortogonales de variación: qué se computa (simulación sobre perfil fijo vs. utilidad esperada cerrada), qué motor de creencias actualiza el posterior (secuencial single-test, conteo exacto, Gibbs) y qué regla de selección (miope vs. lookahead).

### Greedy miope (selector de pool)
- **Ubicación:** `augmented/greedy.py:_myopic_best_pool`
- **Idea / diferencia:** ES la definición del greedy miope. Recorre los pools candidatos y elige el que maximiza Score(t) = ∏(1−p_i)·Σu_i sobre los miembros no cleared, es decir P(r=0) por la utilidad inmediata cosechable. No anticipa pasos futuros. Con `use_filtering=True` restringe candidatos al `active_mask` (`compute_active_mask` con `include_known_healthy=True`), dejando elegibles a los deducidos-sanos para cosechar su utilidad en un pool de r=0 garantizado.
- **Estado:** maduro.
- **Hallazgos:** es la pieza por defecto de todas las demás variantes. La lógica de filtrado de sanos-deducidos se valida en `test_known_healthy_individuals_are_harvested` y `test_eu_still_equals_simulation_after_harvest_fix` (`tests_correctness_fixes.py`). Solo el término r=0 importa para la utilidad inmediata, lo que hace la selección idéntica entre tests clásicos y aumentados.

### Greedy miope (simulación sobre perfil fijo)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_simulate`
- **Idea / diferencia:** realiza la política miope sobre un perfil fijo `z_mask`: en cada uno de los B pasos elige `_myopic_best_pool`, observa `test_result(pool, z_mask)`, marca cleared si r=0 y actualiza con `bayesian_update_single_test`. Acepta `pool_selector` inyectable. Es la misma regla miope materializada sobre un mundo concreto.
- **Estado:** maduro.
- **Hallazgos:** sirve de ground truth de la política miope (`_true_policy_eu` enumera 2^n perfiles y promedia). Cubierto por `test_greedy_simulate_all_healthy`, `test_counting_greedy_matches_sequential_z0` y `test_sequential_eu_equals_policy_simulation`.

### Greedy miope (utilidad esperada cerrada)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_expected_utility`
- **Idea / diferencia:** versión analítica (no simulada) del miope. Recursa sobre el árbol de resultados eligiendo con `_myopic_best_pool` (marginales secuenciales) y ramifica sobre r=0..|t|. Detalle de correctness: los pesos de rama son P(r|history) vía `_branch_pmf`, que usa `exact_pool_pmf` (enumeración exacta) cuando n≤`EXACT_PMF_MAX_N`=18 y cae a Poisson-Binomial para n grande. La selección usa marginales pero las ramas usan la distribución exacta.
- **Estado:** maduro.
- **Hallazgos:** `test_sequential_eu_equals_policy_simulation` verifica que iguala al ground truth (gap < 1e-9) en (4,2,3), (5,3,3), (6,2,3). Aparece en la cadena U_single ≤ U_greedy ≤ U_opt ≤ U_max (`test_greedy_inequality_chain`) y en `test_greedy_myopic_B1` (con B=1 el miope es óptimo, iguala a `solve_optimal_dapts`).

### Lookahead / greedy no-miope (selector con anticipación)
- **Ubicación:** `augmented/greedy.py:_lookahead_best_pool` (apoyado por `_greedy_future`)
- **Idea / diferencia:** el único selector NO-miope. Para cada pool suma sobre r=0..|t|, aplica el update single-test, y en vez de la utilidad inmediata evalúa la utilidad FUTURA esperada de los pasos restantes vía `_greedy_future`; elige el pool que maximiza E[utilidad total] y considera no testear (pool=0). Anticipación parcial: el primer nivel es lookahead completo (prueba TODOS los pools), pero las ramas futuras se evalúan asumiendo política MIOPE. Es aproximado, no el DP óptimo, y usa pesos Poisson-Binomial (`_poisson_binomial_pmf`), no `exact_pool_pmf`.
- **Estado:** experimental.
- **Hallazgos:** solo lo cubre el smoke test `test_greedy_lookahead_simulate` (verifica util > 0 con z=0); no hay test que compruebe que mejora al miope ni que iguale al DP.

### `_greedy_future` (evaluador de continuación miope)
- **Ubicación:** `augmented/greedy.py:_greedy_future`
- **Idea / diferencia:** auxiliar del lookahead. Evalúa la EU de seguir la política MIOPE durante los b tests restantes: selecciona con `_myopic_best_pool`, ramifica con pesos Poisson-Binomial y recursa. Es exactamente un rollout miope usado como estimador de continuación.
- **Estado:** experimental.
- **Hallazgos:** sin test directo propio; se ejercita solo dentro de `_lookahead_best_pool` / `greedy_lookahead_simulate`.

### Lookahead greedy (simulación sobre perfil fijo)
- **Ubicación:** `augmented/greedy.py:greedy_lookahead_simulate`
- **Idea / diferencia:** simula el lookahead sobre `z_mask` fijo. Paso 0 usa `_lookahead_best_pool`; pasos 1+ caen a `_myopic_best_pool` (de lo contrario sería el DP completo). Híbrido: anticipa solo en el primer paso y luego es idéntico al miope.
- **Estado:** experimental.
- **Hallazgos:** solo smoke test `test_greedy_lookahead_simulate`. Se usa como demo en `example.py`; NO está integrado en `comparison.py` ni `experiments.py`. El docstring reconoce explícitamente que el lookahead en todos los pasos sería el DP óptimo, evitado por coste.

### Greedy miope con conteo (simulación)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_counting_simulate`
- **Idea / diferencia:** igual que `greedy_myopic_simulate` pero recomputa el posterior desde la HISTORIA COMPLETA con `bayesian_update_by_counting` (enumera los 2^n perfiles consistentes con toda la historia, no updates secuenciales independientes). El selector sigue siendo miope; captura información cruzada entre tests que el update secuencial pierde.
- **Estado:** maduro.
- **Hallazgos:** `test_counting_greedy_simulate`, `test_counting_greedy_matches_sequential_z0` (coincide con el secuencial cuando z=0), `test_counting_captures_cross_test_info`, `test_counting_full_history_cross_test`, `test_counting_raises_on_infeasible_history`.

### Greedy miope con conteo (utilidad esperada)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_counting_expected_utility`
- **Idea / diferencia:** versión analítica de la EU miope-conteo. Recursa sobre el árbol recomputando el posterior con `bayesian_update_by_counting` y seleccionando con `_myopic_best_pool`. Pesos de rama vía `_branch_pmf` (exacto cuando n≤18). El comentario subraya que usar Poisson-Binomial de marginales sería erróneo porque condicionar en la historia destruye la independencia.
- **Estado:** maduro.
- **Hallazgos:** aquí vivió el **Fix #1**: el código antiguo usaba Poisson-Binomial de marginales y SOBREESTIMABA su propia política (caso auditoría p=[0.45]×5, u=[2,2,2,1,1], B=4, G=3: EU buggy 3.629935 vs. real 3.098026, +17%). Ahora `test_counting_eu_equals_policy_simulation_audit_case` y `*_random_instances` verifican gap < 1e-9 contra `_true_policy_eu`. También `test_counting_greedy_expected_utility` y `test_counting_vs_sequential_greedy_eu`.

### Greedy miope con Gibbs (simulación)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_gibbs_simulate`
- **Idea / diferencia:** igual que la variante de conteo pero aproxima el posterior de la historia completa con muestreo de Gibbs (`gibbs_update`, num_iterations=1000, burn_in=200) en vez de enumeración exacta. Pensado para escalar a n~50+ donde el conteo O(2^n) es infeasible. Selector miope.
- **Estado:** experimental.
- **Hallazgos:** `test_gibbs_greedy_simulate` (smoke), `test_gibbs_approx_matches_counting`, `test_gibbs_systematic_exact_comparison`. El Gibbs tuvo un fix de validez (contaba perfiles inválidos en el camino >7 activos; stopgap 2026-06-03, reescritura de mixing aún abierta). Ergodicidad cubierta por `test_gibbs_mcmc_is_ergodic_across_count_levels` y `test_gibbs_components_solved_independently`.

### Greedy miope con Gibbs (utilidad esperada)
- **Ubicación:** `augmented/greedy.py:greedy_myopic_gibbs_expected_utility`
- **Idea / diferencia:** versión analítica de la EU miope-Gibbs. Recursa recomputando el posterior con `gibbs_update` (seed fijo=42 para reproducibilidad dentro del árbol) y seleccionando con `_myopic_best_pool`. Pesos de rama vía `_branch_pmf` (exacto cuando n≤18).
- **Estado:** experimental.
- **Hallazgos:** `test_gibbs_greedy_expected_utility` y `test_gibbs_greedy_vs_counting_eu` comparan contra la EU de conteo en instancias pequeñas; deben quedar cerca.

## 2. Formulación VW de super-nodos (propuesta de Francisco Marmolejo)

Tres scripts que comparten el hilo Q3 de `docs/notes/vw_submodularity.md`. Conclusión central verificada al correr el código: el VW ESCALAR con prob all-clear reproduce EXACTAMENTE el paso del greedy miope; no es una mejora algorítmica sino una reformulación. El único valor práctico hallado es de tractabilidad (poda top-L).

### Demo de las tres formulaciones (full / VW-A all-clear / VW-B or-event)
- **Ubicación:** `augmented/vw_demo.py:_enumerate_full_pools`, `_enumerate_vw`, `main`
- **Idea / diferencia:** tras k tests construye S=unión de pools, V=N\S y plantea el paso siguiente con super-nodos escalares w_T (peso=|T|, prob, util_T=Σu_i). El pool es t=U∪T con |t|≤G y a lo sumo un w_T. Puntúa (util(U)+util_T)·pclear(U)·prob_T con dos lecturas: all-clear ∏(1−p_i) (modo A) y or-event 1−∏(1−p_i) (modo B). Reorganiza el cálculo en super-nodos en vez de enumerar pools planos, pero NO añade lookahead.
- **Estado:** demo.
- **Hallazgos:** con n=6, G=3, B=2, tras test {0,2,4} con r=1: full enum y VW-A eligen el MISMO pool {1,3,4} val=10.2176 (igualdad exacta); VW-B elige distinto y peor (10.0737). End-to-end: B=2 greedy(true)=18.4414, DP=18.5817, gap=0.1403; B=3 greedy=20.4740, DP=22.4339, gap=1.9599 → el VW escalar vive en la línea greedy(true) y no cierra el gap. Counterexample: T1={a} p=.5 y T2={b,c} p=.2929 tienen MISMA or-prob y all-clear pero PMF de conteo distinta (H: 1.0 vs 1.3306 bits): ningún escalar los distingue, solo la PMF completa.

### Experimento de restricción top-L (L_min por heurística de ranking)
- **Ubicación:** `augmented/vw_restrict.py:restriction_experiment`, `_best_pool_with_T`, `L_min`, `h_self/h_prob/h_util/h_partner/make_h_entropy/h_rand`, `run_trial`, `adversarial_instance`, `main`
- **Idea / diferencia:** Q3: W tiene 2^|S|−1 super-nodos; ¿se puede quedar con un top-L barato sin perder el T óptimo? Define val(T)=mejor pool T∪U con |t|≤G y mide L_min(rank)= menor L tal que el top-L contiene un T óptimo. Heurísticas O(|T|): self=(Σu_T)·∏(1−p_T), prob, util, ent_λ=self+λ·H(r_T), partner=(Σu_T+u*)·∏(1−p_T)·p*. Maquinaria para hacer tractable la enumeración del paso miope reformulado, sin tocar el objetivo.
- **Estado:** experimental.
- **Hallazgos:** n=10, G=4, K=20, mean|W|=60.5: mean/max L_min partner 1.05/2, self 3.40/17, ent_1 3.65/17, prob 20.35/55, util 15.70/50, rand 26.45/88. Adversarial (S={0,1,2,3}, V={4,5}, G=4): óptimo {3,4} val=99.275 vs val_empty=66.5; L_min partner=2 (13%), self=8 (53%). partner domina; self_score falla porque rankea {2,3} arriba (util_T=150 con prob_T=0.4).

### Barrido de regímenes (robustez de partner vs. self)
- **Ubicación:** `augmented/vw_restrict_sweep.py:run_regime`, `_gen_population`, `REGIMES`, `main`
- **Idea / diferencia:** corre `run_trial` K=20 veces en 6 regímenes (baseline, alta prevalencia, n grande, historia profunda, bimodal, utilidad atípica) y agrega mean/max de L_min. Reusa el paso miope (val(T) por score ∏(1−p)·Σu); no introduce ningún criterio nuevo de decisión.
- **Estado:** experimental.
- **Hallazgos:** partner domina en 5 de 6 regímenes y su mean L_min se mantiene ~1-3 aunque |W| crece de 79 a 278 (escala independiente de |S|). Único régimen donde self gana: alta prevalencia (óptimo casi siempre |T|≤1). ent_λ NO ayuda. Sin cota de peor caso: max L_min de self llega a 132 (~47% de |W|) en n=15.

> Lo abierto en esta familia: (1) no hay cota de peor caso para L_min; (2) ent_λ se probó y no ayuda; (3) el problema teórico real es Q4 — si el conteo r preserva submodularidad adaptativa (Golovin-Krause). Para cerrar el gap de lookahead haría falta cargar la PMF de conteo completa en w_T, momento en que el VW deja de ser super-nodo escalar y vuelve a ser la enumeración DP existente. Los notebooks `06_vw.ipynb` y `07_vw.ipynb` son write-ups de estos scripts (07 además arranca pruebas de RL tabular).

## 3. El independence gap y el greedy de scoring EXACTO (`independence_gap.py`)

El módulo tiene dos mitades: medición del error de aproximar la conjunta del pool por producto de marginales, y un payoff (greedy miope con scoring exacto). La diferencia esencial con el miope: el producto de marginales se reemplaza por la probabilidad conjunta exacta enumerada sobre los perfiles aún consistentes.

### `exact_pool_pmf` (PMF posterior exacta del conteo)
- **Ubicación:** `augmented/bayesian.py:214`; importada en `augmented/independence_gap.py:39-43,77`
- **Idea / diferencia:** calcula la PMF posterior VERDADERA de r_t dado el historial enumerando los 2^n perfiles, quedándose con los consistentes y agregando pesos del prior por conteo. El miope NO calcula esto: aproxima la conjunta como producto de marginales. Aquí puede dar P(r_t=0|H)=0 cuando un subconjunto t' ya salió positivo.
- **Estado:** maduro.
- **Hallazgos:** `test_exact_pool_pmf_empty_history_matches_poisson_binomial` y `test_exact_pool_pmf_singleton_gives_marginal` (`tests.py:922-943`). Caro: O(2^n), para n≤8/14. SÍ se usa en producción dentro de `greedy.py` (`_branch_pmf`, vía `EXACT_PMF_MAX_N`=18) para pesar las ramas de la EU recursion.

### `independence_pool_pmf` (PMF heurística producto-de-marginales)
- **Ubicación:** `augmented/independence_gap.py:51-61`
- **Idea / diferencia:** devuelve la PMF de r_t bajo independencia: toma los marginales posteriores exactos y construye la Poisson-Binomial. Es exactamente la distribución implícita del miope, materializada como vector completo (no solo el endpoint r=0).
- **Estado:** maduro.
- **Hallazgos:** lado heurístico de todas las comparaciones; sin hallazgos propios más allá de servir de referencia en `gap_summary`.

### `tv_distance` + `gap_summary` (métricas de la brecha)
- **Ubicación:** `augmented/independence_gap.py:64-89`
- **Idea / diferencia:** `tv_distance` da la distancia de variación total entre dos PMFs. `gap_summary` reporta tv, gap_r0 = heur[0]−exact[0] (el endpoint que conduce el scoring del miope) y gap_rmax. Cuantifica cuánto se equivoca el miope al multiplicar marginales.
- **Estado:** maduro.
- **Hallazgos:** caso trabajado: t'={0,1}, r'=1, t={0,1,2,3}, prior 0.5: exact P(r=0)=0.0, heur=0.0625, TV=0.1250. Tests `test_singleton_gap_is_zero`, `test_deterministic_subset_shows_gap`, `test_all_healthy_subset_heuristic_is_exact` (`tests.py:946-989`).

### `run_experiment` + `aggregate` (barrido de la brecha)
- **Ubicación:** `augmented/independence_gap.py:108-184`, `284-306`; driver `augmented/independence_gap_demo.py`
- **Idea / diferencia:** muestrea priors y perfiles, genera historiales (greedy/random/none) y registra `gap_summary` por pool; `aggregate` resume por tamaño de pool. Cuantifica cuán grande es la brecha típicamente, no solo en el caso patológico.
- **Estado:** maduro.
- **Hallazgos:** n=8, B=3, G=3, 300 instancias, historial greedy: |t|=2 → TV media 0.0243, mediana 0.0, p95 0.2166, max 0.5000; |t|=3 → TV media 0.0501, mediana 0.0, p95 0.4696, max 0.5555. **Lectura central:** la brecha es CERO en la mediana (la heurística suele acertar) con cola larga grande cuando un subconjunto ya salió positivo.

### `exact_greedy_myopic_expected_utility` (+ `_exact_best_pool`, `_prior_weights_indep`)
- **Ubicación:** `augmented/independence_gap.py:199-281`
- **Idea / diferencia:** greedy miope idéntico en estructura pero que puntúa cada pool con la probabilidad de clearing EXACTA P(r_t=0|H) en vez del producto de marginales. `_exact_best_pool` mantiene el conjunto `remaining` de perfiles consistentes y calcula prob_clear como masa de perfiles con test_result=0. Es la corrección directa de `_myopic_best_pool`.
- **Estado:** experimental.
- **Hallazgos:** `test_exact_greedy_matches_heuristic_on_trivial` (n=1) y `test_exact_greedy_bounded_by_optimal` (EU ≤ óptimo del DP) (`tests.py:1000-1016`). Es la prueba de concepto de que se PUEDE usar scoring exacto. Costo O(2^n·#pools·B·estados), solo n≤8.

> Lo a medias (`estado_proyecto_2026-05.md` §3.2): ya se midió la brecha y existe el greedy exacto para n chico, pero el greedy de producción sigue usando la multiplicación. Falta (a) un barrido comparativo de utilidad exact-greedy vs. miope estándar para decidir si la brecha cambia decisiones, y (b) una versión del scoring exacto que escale más allá de n≤8.

## 4. Greedies de recompensa alternativa y solvers híbridos

Tres piezas que comparten ADN con el miope y reutilizan su recursión para EU exacta, pero intervienen en lugares distintos: el SCORE de selección (beta), la DEFINICIÓN de recompensa (alpha), o el cierre del horizonte (DP).

### Greedy de infection-reward (beta-greedy)
- **Ubicación:** `augmented/infection_reward_greedy.py:_beta_best_pool`, `_compute_info_gain`, `greedy_myopic_beta_simulate`, `greedy_myopic_beta_expected_utility`; helpers `run_vip_benchmark`, `run_beta_sweep`
- **Idea / diferencia:** elige el pool que maximiza un score de DOS términos: el miope clásico P(r=0)·Σu MÁS beta·E[ganancia de información]. La ganancia (`_compute_info_gain`) se calcula exacto sobre la PMF Poisson-Binomial; tres métricas seleccionables (`info_metric`): 'entropy', 'variance', 'confirmed'. El miope solo valora limpiar utilidad ahora; este agrega un premio explícito por APRENDER quién está infectado, inclinando hacia pools más grandes y diagnósticos. Generalización estricta: con beta=0 reproduce el miope.
- **Estado:** maduro.
- **Hallazgos:** tests dedicados 5/5 (`tests_infection_reward_greedy.py`), incluido `test_beta_zero_matches_standard_greedy_eu`. Registrado en `phase3_findings.ipynb`: (1) en alta prevalencia p_vip=0.8 beta NO cambia nada porque P(r=0)=0.2^k colapsa y el término miope domina; (2) en prevalencia moderada p_vip=0.35 beta=1.0 desplaza el primer pool de tamaño ~2 a 5+ y mejora ligeramente la EU. **Recomendación:** beta es valioso solo en prevalencia moderada (p~0.2-0.4). Usado en `sprint3_experiments.py`. Inconsistencia menor: default `info_metric='entropy'` pero `run_vip_benchmark` usa 'confirmed'.

### Greedy de semi-utilidad (alpha-blended)
- **Ubicación:** `augmented/semi_utility.py:semi_utility`, `_semi_best_pool`, `greedy_myopic_semi_simulate`, `greedy_myopic_semi_expected_utility`
- **Idea / diferencia:** cambia la DEFINICIÓN de recompensa, no la mecánica. U_semi = Σ_i u_i·[alpha·P(sano_i|H) + (1−alpha)·1_{limpiado}(i)]. Con alpha=0 es el modelo binario clásico; con alpha=1 la utilidad es proporcional a la probabilidad posterior de estar sano. El miope solo premia el evento r=0 (limpieza dura); la semi-utilidad también recompensa empujar posteriors hacia sano (limpieza blanda), por lo que `_semi_best_pool` integra sobre TODOS los r posibles. La utilidad FINAL reportada para comparación sigue siendo la binaria estándar.
- **Estado:** experimental.
- **Hallazgos:** NO hay archivo de tests dedicado. El módulo corre e importa limpio; se importa en `build_paper_notebook.py`, `paper_findings.ipynb` y `examples_notebook.ipynb`, pero NO aparece como barra propia en las figuras finales del paper. En `estado_proyecto_2026-05.md` figura como implementado pero placeholder exploratorio ('no cubre falsos positivos/negativos', lo único parecido a la dirección de '4 utilidades por persona'). Soporta update_method sequential/counting/gibbs. No hay hallazgo numérico de que mejore al miope.

### Solver híbrido greedy→DP exacto
- **Ubicación:** `augmented/hybrid_solver.py:hybrid_greedy_bruteforce` (entrada), `_hybrid_recurse`, `_dp_phase`, `_full_greedy_tree`, `_greedy_fallback`; scoring `infection_aware_score`, `expected_info_gain`, `_infection_aware_best_pool`; cotas `estimate_branch_value`
- **Idea / diferencia:** resuelve los primeros K=greedy_steps pasos con greedy (por defecto el miope, o inyectable) y los B−K restantes con DP EXACTO (`solve_optimal_dapts`). Clave para tractabilidad: `_dp_phase` REDUCE el problema a solo los agentes activos (`compute_active_mask` quita limpiados y confirmados), corre el DP sobre ese subproblema y REMAPEA los índices de vuelta, sumando la utilidad ya asegurada. Si quedan >14 activos cae a greedy continuado. Es un PUENTE miope↔óptimo: barriendo K de B a 0 interpola entre greedy puro y DP puro.
- **Estado:** maduro.
- **Hallazgos:** 14/14 tests (`tests_hybrid.py`): `test_hybrid_k0_matches_dp` (K=0 reproduce el DP), `test_hybrid_kB_matches_greedy` (K=B reproduce el greedy), `test_hybrid_monotonic` (EU monótona creciente en K), validación de `infection_aware_score` (alpha=1 == miope, alpha=0 == info-gain). Probado con selector MOSEK (`test_hybrid_with_mosek_pool_selector`). Usado en `large_trees_exploration.ipynb` para medir el gap de optimalidad. Por construcción su EU domina (≥) la del miope para K>0.

> Solapamiento real de scoring: `infection_reward_greedy._compute_info_gain` y `hybrid_solver.expected_info_gain`/`infection_aware_score` son implementaciones casi paralelas del mismo concepto (hybrid usa log2 en bits, infection_reward usa log natural), en módulos distintos sin compartir código. Madurez dispar: beta-greedy e híbrido tienen tests y hallazgos; semi_utility quedó a medias (sin tests, sin hallazgo cuantitativo, ausente de las figuras finales), antesala de las direcciones aún no empezadas (falsos positivos/negativos, '4 utilidades por persona'). Hallazgo conceptual más sólido: el premio por información solo mueve la aguja en prevalencia moderada.

## 5. Selección de pool por optimización y solvers exactos

Dos subgrupos. (a) Solvers de UN PASO (`pool_solvers.py`): drop-in del miope que resuelven el mismo argmax por optimización en vez de enumeración. (b) Solvers GLOBALES/exactos: no son variantes del greedy sino los TECHOS y PISOS de la cadena U_single ≤ U_s_NO ≤ U_s_O ≤ U_D ≤ U_D_A.

### `mosek_best_pool`
- **Ubicación:** `augmented/pool_solvers.py:mosek_best_pool`
- **Idea / diferencia:** resuelve EXACTAMENTE el mismo argmax de un paso que el miope, formulado con cono exponencial (Mosek Fusion). Variables binarias x_i seleccionan miembros; el cono (z,1,y)∈K_exp impone y≤log(z); objetivo max y+Σx_i·log(q_i). Restricción 1≤Σx_i≤G, mioMaxTime 30s, MIPGap 1e-3; cae a `_heuristic_best_pool` si falla la licencia. No cambia la política, solo cómo se encuentra el argmax.
- **Estado:** maduro.
- **Hallazgos:** mismo score que la enumeración bruta para n=5 y en 10 instancias aleatorias (tol 1e-4); concuerda con gurobi. Benchmarks: n=30 G=5 <10s, n=50 G=5 <30s; greedy completo n=30 B=2 G=5 <120s. **Sutileza:** usa `compute_active_mask(include_known_healthy=False)`, mientras `_myopic_best_pool` usa `True`, así que difieren en estados con sanos deducidos (sin test que lo cubra).

### `gurobi_best_pool`
- **Ubicación:** `augmented/pool_solvers.py:gurobi_best_pool`
- **Idea / diferencia:** gemelo de mosek en Gurobi: MILP con x_i binarias, z=Σu_i·x_i, y=log(z) impuesto con `addGenConstrLog`. Mismo objetivo. TimeLimit 30s, MIPGap 1e-3; acepta OPTIMAL/SUBOPTIMAL/TIME_LIMIT con incumbente; cae a heurístico ante fallo.
- **Estado:** maduro.
- **Hallazgos:** concuerda con enumeración bruta (n=5) y con mosek en score. Benchmarks idénticos. Solver por defecto del runner `overnight_experiments.py` (--solver gurobi). Misma sutileza `include_known_healthy=False`.

### `_heuristic_best_pool`
- **Ubicación:** `augmented/pool_solvers.py:_heuristic_best_pool`
- **Idea / diferencia:** atajo greedy que ordena los activos por u_i·(1−p_i) y toma los G mejores. NO maximiza el producto P(r=0)·Σu (no separable); es una aproximación separable por individuo, aún más miope que el miope. Solo se usa cuando Mosek/Gurobi fallan o no tienen licencia.
- **Estado:** maduro.
- **Hallazgos:** ninguno de calidad propia; red de seguridad. No hay test que mida su gap respecto al óptimo. Puede diferir del miope porque ignora la interacción multiplicativa de q_i dentro del pool.

### `solver_best_pool`
- **Ubicación:** `augmented/pool_solvers.py:solver_best_pool`
- **Idea / diferencia:** despachador trivial que enruta a mosek o gurobi según `solver`; lanza ValueError para otros. No añade lógica de optimización.
- **Estado:** maduro.
- **Hallazgos:** ninguno. Pasarela para usar cualquiera de los dos solvers exactos como `pool_selector`.

### `solve_optimal_dapts`
- **Ubicación:** `augmented/solver.py:solve_optimal_dapts` (con `_MAX_N`=14)
- **Idea / diferencia:** óptimo GLOBAL de la política dinámica adaptativa augmented por DP sobre estados (k, remaining_set, cleared_mask). Prueba TODOS los pools y particiona los perfiles por el resultado EXACTO de conteo r (factor de ramificación |pool|+1); reconstruye la política DAPTS óptima desde los argmax. Es el techo contra el que se mide el greedy: resuelve TODO el horizonte con anticipación perfecta y observación exacta del conteo.
- **Estado:** maduro.
- **Hallazgos:** es el U_D_A en `comparison.py` y la referencia de oro en `cross_verification.py` (exacto solo n≤14). Límite codificado n≤14, pero el límite práctico real registrado en memoria es N=G=5, B=3. El gap U_D_A − U_greedy cuantifica la pérdida por miopía.

### `solve_classical_dynamic`
- **Ubicación:** `augmented/classical_solver.py:solve_classical_dynamic` (`_MAX_N`=14)
- **Idea / diferencia:** misma estructura DP pero con resultado BINARIO: cada pool particiona en negativo (despeja) y positivo, factor de ramificación 2 en vez de |pool|+1. Óptimo global de la política dinámica clásica. Devuelve solo el valor.
- **Estado:** maduro.
- **Hallazgos:** es el U_D de la cadena. La diferencia U_D_A − U_D mide el beneficio del conteo exacto sobre el test binario; el hallazgo de horizonte indica que ese beneficio es fenómeno del horizonte B (B=1 ⇒ beneficio 0).

### `solve_static_non_overlapping`
- **Ubicación:** `augmented/static_solver.py:solve_static_non_overlapping` (`_MAX_N`=14)
- **Idea / diferencia:** U^s_NO: enumera recursivamente todas las formas de elegir B pools DISJUNTOS de tamaño ≤G y maximiza Σ_k P(pool_k todo sano)·Σu. Estático: los B pools se fijan de antemano, sin adaptar. Baseline (no greedy): el miope SÍ adapta paso a paso, este no.
- **Estado:** maduro.
- **Hallazgos:** primer eslabón estático tras U_single. Límite n≤14.

### `solve_static_overlapping`
- **Ubicación:** `augmented/static_solver.py:solve_static_overlapping` (`_MAX_N`=14)
- **Idea / diferencia:** U^s_O: enumera todas las B-tuplas de pools (permitiendo solape) y evalúa por fuerza bruta sobre los 2^n perfiles; i se despeja si EXISTE algún pool que lo contiene con resultado negativo. Baseline estático sin adaptación; el solape lo hace más fuerte que U_s_NO.
- **Estado:** maduro.
- **Hallazgos:** es el U_s_O (U_s_NO ≤ U_s_O). Límite n≤14; costo extra por evaluar 2^n perfiles por asignación.

> Nada quedó a medias en esta familia: todo tiene tests (`tests_solvers.py`) y está cableado en `comparison.py` / `cross_verification.py` / `overnight_experiments.py`. Lo único a vigilar es la sutileza de `include_known_healthy=False` en los solvers de un paso, que los aparta levemente del miope canónico en estados con sanos deducidos.

## 6. Aprendizaje por refuerzo (RL)

Dos sublíneas. (a) RL pedagógico-exacto (`rl_examples.py`): maduro, validado contra el DP; sirve de cota superior, no extiende al greedy. (b) RL profundo PPO (`rl_env.py` + `rl_train.py`): experimental, con modelos `.zip` de demostración pero SIN métricas registradas. `DaptsBucketEnv` es el único punto donde RL se hibrida realmente con el greedy.

### `DaptsExactEnv` (entorno de estado de creencia exacto)
- **Ubicación:** `augmented/rl_env.py:DaptsExactEnv` (reset, _obs, step)
- **Idea / diferencia:** MDP de creencia exacta para n pequeño. Observación: posterior completo sobre los 2^n perfiles + indicador de cleared + fracción de presupuesto. Acción: pool concreto de `all_pools(n,G)` (espacio Discrete). Recompensa terminal en el paso B = utilidad de los probados sanos. A diferencia del miope, NO elige localmente: la política entrenada puede en principio aprender el óptimo del horizonte completo porque ve el posterior exacto.
- **Estado:** experimental.
- **Hallazgos:** modelo PPO guardado `augmented/rl_models/exact_n3_B2_G3_s0.zip`. La validación `evaluate_exact_vs_dp` (`rl_train.py`) compara el valor exacto de la política PPO contra `solve_optimal_dapts`, pero NO hay número de cociente registrado en el repo. Banco de pruebas para ver si RL recupera el óptimo, sin métricas guardadas.

### `DaptsBucketEnv` (RL elige el primer pool + greedy juega el resto)
- **Ubicación:** `augmented/rl_env.py:DaptsBucketEnv` (_category, reset, _obs, step, _rollout_reward)
- **Idea / diferencia:** entorno para N grande (más allá del muro DP n≤14). Observación: histograma de tamaño fijo por (cubeta de salud × cubeta de utilidad) + agregados de lo seleccionado → escala en N. Acción: elegir un agente de una categoría o STOP; RL ENSAMBLA solo el PRIMER pool. Recompensa: se muestrea z, el test 1 es el pool de RL y los tests 2..B los juega el greedy miope augmentado (`greedy_myopic_simulate` con `pool_selector` que devuelve el pool RL en la primera llamada y `_myopic_best_pool` después). Descendiente directo del greedy: RL decide el primer pool y delega el resto.
- **Estado:** experimental.
- **Hallazgos:** modelo PPO guardado `augmented/rl_models/bucket_N50_B2_G3_s0.zip`. `evaluate_bucket` compara en episodios emparejados PPO vs. greedy puro e imprime la diferencia, pero NO hay valor registrado. Auditoría 2026-06-09 (`tests_rl_fixes.py`) corrigió dos bugs de observación/binning (usum podía superar N; `utility_bin_edges` fijo ignorando el argumento), ambos con tests de regresión que pasan.

### `value_iteration` (inducción hacia atrás sobre el MDP)
- **Ubicación:** `augmented/rl_examples.py:value_iteration` (+ `value_iteration_optimal_value`, `_prior_weights`, `_cleared_utility`, `_transition`)
- **Idea / diferencia:** reescribe el DP exacto en vocabulario de MDP de horizonte finito: estado (k, remaining, cleared_mask), transición por r=|a∩Z|, recompensa terminal. Barrido Bellman hacia atrás → V*, Q*, π*. Matemáticamente idéntico a `solve_optimal_dapts`; puente didáctico DP↔value iteration. No es greedy: mira todo el horizonte.
- **Estado:** maduro.
- **Hallazgos:** validado contra el DP: `test_vi_matches_dp_small` (n=2,B=1, <1e-12), `test_vi_matches_dp_medium` (n=3,B=2, <1e-10), `test_vi_matches_dp_several_randoms` (10 instancias, <1e-9), `test_vi_returns_valid_policy`.

### `tabular_q_learning` (Q-learning tabular, model-free)
- **Ubicación:** `augmented/rl_examples.py:tabular_q_learning` (+ `_sample_profile`, `q_learning_policy_value`)
- **Idea / diferencia:** Q-learning epsilon-greedy sobre el mismo MDP de información, sin asumir conocer el modelo: muestrea z por episodio, interactúa B pasos y actualiza Q(s,a) por TD(0). Con alpha='auto' usa Robbins-Monro 1/(1+N(s,a)). A diferencia del miope (regla fija sin aprendizaje), la política se APRENDE y converge al Q* del horizonte completo.
- **Estado:** maduro.
- **Hallazgos:** `test_q_learning_recovers_optimal_tiny` (n=2,B=1, 3000 episodios, <1e-9) y `test_q_learning_near_optimal_medium` (n=3,B=2, 20000 episodios, ≥5 de 8 semillas dentro del 5%). El demo traza la curva de aprendizaje (`figures/rl_q_learning_curve.png`). Q-learning converge al óptimo del DP en instancias chicas.

### `AgentSelectionEnvB2/B3/B4/B5` (entornos PPO clásicos por cubetas, con MOSEK)
- **Ubicación:** `classical/rl_training/PPO_bucket_gymnasium_B2.py:AgentSelectionEnvB2`; B3/B4/B5 heredan y sobreescriben `_compute_reward`
- **Idea / diferencia:** esquema anterior del que `DaptsBucketEnv` es el puerto. Misma observación por histograma; RL ensambla el primer pool. Diferencia clave: los tests posteriores NO los juega el greedy miope, sino un solver cónico de un pool (`solveConicSingle`, MOSEK) con Bayes exacto. Diseño ANIDADO: B3 llama recursivamente a B2, B4 a B3, B5 a B4 — un PPO por nivel de presupuesto apilado.
- **Estado:** abandonado.
- **Hallazgos:** dependen de MOSEK y de modelos `.zip` por nivel B que NO están en el paquete augmented. `rl_evaluation/PPO_bucket_gymnasium_use.py` orquesta B2..B5 sobre CSV pero sin métricas resumidas. Legacy: el augmented lo reemplaza quitando MOSEK y usando greedy en el rollout.

> Lo a medias: existe la infraestructura de evaluación (`evaluate_exact_vs_dp`, `evaluate_bucket`) pero NO hay corridas con números guardados de PPO vs. DP ni PPO vs. greedy. La línea VW/super-nodos de Marmolejo es una vía de garantías distinta de esta familia RL, aunque comparta la motivación de mejorar sobre el miope.

## 7. Esquema anterior (`classical`) y algoritmos solo-en-notas

Distinción código-vs-nota clara. CÓDIGO QUE CORRE en `classical/solvers/`: dos greedy dinámicos, el MILP estático, el DP óptimo, el selector miope cónico y baselines estáticos (`milpSample.py` y `greedyDynamicSample.py` son casi idénticos, scripts-notebook exportados con `# %%`, no un paquete modular). SOLO NOTAS: la garantía (1−1/e), la restricción VW por ranking y la selección por Mosek/Gurobi para n grande.

### Greedy Dinámico Clásico (conic + Gibbs, árbol completo)
- **Ubicación:** `classical/solvers/milpSample.py:solveConicGibbsGreedyDynamic` (1060-1173); usa `solveConicSingle` (294) y `GibbsMCMCWindow` (558)
- **Idea / diferencia:** recalcula marginales con Gibbs sobre el historial (posGroups, negAgents), elige UN pool con `solveConicSingle` (MOSEK, cono exponencial) y RAMIFICA recursivamente en escenario positivo y negativo, acumulando utilidad por toda la budget B. Difiere del miope en tres ejes: (1) resuelve el argmax con programa cónico, no enumeración; (2) usa Gibbs/Bayes sobre historial, no el prior; (3) construye el árbol binario completo, no una trayectoria miope. Es el ancestro directo del greedy reimplementado en augmented.
- **Estado:** experimental.
- **Hallazgos:** registrados en docs (no in-situ): `benchmark_tesis_papers.md` cita greedy dinámico vs. MILP en N=50 con +2.76% bienestar (G=5) y +1.24% (G=3), ganando al MILP en 53.87% de casos; ≈99.0% del overlapping en N=G=3,B=2 y >99.5% en N=G=5,B=3; gap greedy 1.5 vs. óptimo 1.75. Limitación declarada: usa solo marginales (no la conjunta), puede calcular mal P(test negativo).

### Greedy Dinámico Clásico con Conteos (count-aware, una sola trayectoria)
- **Ubicación:** `classical/solvers/greedyDynamicSample.py:solveConicGibbsGreedyDynamicCount` (982-1006); usa `GibbsMCMCWindowCount` (344) y `solveConicSingle` (294)
- **Idea / diferencia:** en cada paso recomputa marginales con `GibbsMCMCWindowCount`, que respeta restricciones de CONTEO EXACTO (posGroups = (conjunto, conteo)), elige el pool con `solveConicSingle`, OBSERVA el conteo real y condiciona los posteriors. No ramifica: sigue una única trayectoria condicionada al estado real, por eso es más barato. Difiere del miope en que explota la señal de conteo r=|t∩Z| (no solo r=0 vs r>0). Es la versión count-aware del greedy dinámico clásico.
- **Estado:** experimental.
- **Hallazgos:** el propio comentario (línea 981) advierte: 'utility values erroneous from Gibbs sampling and only using marginal probabilities'. Relacionado con el bug de validez de Gibbs (`gibbs_validez_2026-06-03.md`: contar perfiles inválidos en la ruta de conteos exactos). Sin números de bienestar registrados in-situ.

### MILP de asignación (cluster-based, Gurobi)
- **Ubicación:** `classical/solvers/milpSample.py:solveMILP` (345-556), con `approx_model`, `linearise`, `optimal_partition`, `partition`, `delta`, `compute_error`
- **Idea / diferencia:** resuelve de un solo golpe la asignación ESTÁTICA non-overlapping de B tests a clusters, maximizando Σ_t exp(log(Σu·x)+Σx·log(q)). Linealiza el log (variables zind) y aproxima exp con función lineal-por-tramos de K=20 segmentos cuya partición se optimiza por error. No es dinámico ni adaptativo. Difiere del miope en que es un óptimo global (aprox.) de toda la cartera, no una selección secuencial. Es el primo no-adaptativo del miope.
- **Estado:** experimental.
- **Hallazgos:** adaptado de csef/optimisation (edwinlock). MIPGap=0.01. `benchmark_tesis_papers.md`: el percentil 20 del greedy cae por debajo del MILP para B∈{3,4,5},G=5 (welfare cero 0.0278% greedy vs 0.0124% MILP); garantía teórica 1−Δ·B del óptimo.

### DP óptimo dinámico exacto (`solveDynamic`)
- **Ubicación:** `classical/solvers/milpSample.py:solveDynamic` (752-827) y `greedyDynamicSample.py:solveDynamic` (671-746); evaluadores `analyzeTree`/`analyzeTreeGibbs`/`analyzeTreeSample`
- **Idea / diferencia:** calcula la DAPTS óptima exacta: enumera todos los pools ≤G y evalúa recursivamente el bienestar esperado ramificando pos/neg con `bayesTheorem`, tomando el argmax global del árbol. Es el techo U_A^D. No es greedy.
- **Estado:** experimental.
- **Hallazgos:** `benchmark_tesis_papers.md`: factible solo N≤10,G≤10,B≤5; data a gran escala solo N=G=5,B=3; complejidad O(S^B·B·G·2^(B(G+1))). Existe versión re-implementada y verificada en `augmented/solver.py` (`solve_optimal_dapts`, n≤14), documentada en `solver_context.md`. El gap greedy-vs-óptimo (1.5 vs 1.75) se mide contra este DP.

### Selector miope exacto de un test (`solveConicSingle` / MICOP)
- **Ubicación:** `classical/solvers/milpSample.py:solveConicSingle` (294-342); duplicado en `greedyDynamicSample.py:294` y `trial.py:5`
- **Idea / diferencia:** resuelve exactamente el problema de UN test: elige el pool ≤G que maximiza log(P(r=0)·Σu) = Σx·log(q)+log(Σu·x) vía cono exponencial primal (`Domain.inPExpCone`). ES el greedy miope, pero resuelto óptimamente con optimización en vez de enumeración. Es el corazón compartido de toda la familia greedy dinámica clásica.
- **Estado:** experimental.
- **Hallazgos:** garantía heredada (benchmark): conic single-test ≥1−1e−7 del óptimo single-test (FPTAS, Goldberg-Rudolf), O(N^5). `trial.py` es solo un smoke-test (3 agentes u=1, q=0.5, G=2).

### Baselines estáticos (NoPool, NonOverlap, Overlap)
- **Ubicación:** `classical/solvers/{milpSample.py,greedyDynamicSample.py}`: `solveStaticNoPool` (142), `solveStaticNonOverlap` (152), `solveStaticOverlap` (590/671)
- **Idea / diferencia:** tres baselines no-adaptativos. NoPool prueba a los top-B por u·q (individual); NonOverlap enumera B subconjuntos disjuntos ≤G maximizando Σ groupHealthy·groupUtility; Overlap enumera subconjuntos que solapan y evalúa con Bayes por ramas. Difieren del miope en que son estáticos y enumerativos. El miope dinámico los domina débilmente.
- **Estado:** experimental.
- **Hallazgos:** forman la cadena U^single ≤ U_NO^s ≤ U_O^s ≤ U^D ≤ U_A^D ≤ U^max. Tabla 6.1 (N=G=5,B=3): Non-Pooled 1.10, Greedy Non-Ov 1.03, Non-Ov 1.13, Overlapping 1.14, Greedy Dynamic 1.13, Optimal Dynamic 1.15.

### Garantía (1−1/e) vía submodularidad adaptativa (Golovin–Krause)
- **Ubicación:** `docs/notes/vw_submodularity.md` (Q1-Q4 y 'Verdict')
- **Idea / diferencia:** pregunta central de Marmolejo: ¿el greedy dinámico con observación de CONTEO admite una garantía (1−e^{−α}) tipo Golovin-Krause? Define F_h(T,U), enuncia monotonicidad (AM) y submodularidad adaptativa (AS), y argumenta dónde el VW de super-nodos se rompe: el pool es selección en bloque (knapsack, no item-a-item), F_h no es separable, y AS no se sigue de submodularidad por-paso. Q4 (la 'killer question'): ¿el conteo r preserva AS siendo un coarsening de la realización individual?
- **Estado:** solo-notas (NO implementado).
- **Hallazgos:** diagnóstico clave: el VW escalar con prob_A reproduce EXACTAMENTE el greedy miope (re-escritura, no mejora); cerrar el gap requiere la PMF de conteo, momento en que la formulación vuelve a ser la DP completa. El gap miope-vs-óptimo CRECE con B (0.14 en B=2 → 1.96 en B=3, n=6 G=3). `benchmark_tesis_papers.md` confirma: NO hay submodularidad ni garantía (1−1/e) en ninguno de los tres papers; el hilo queda abierto.

### Restricción VW de super-nodos por ranking (self_score / partner / entropy)
- **Ubicación:** `docs/notes/vw_submodularity.md` (Q3 empírico, tablas L_min); código companion en `augmented/vw_restrict.py`, `vw_restrict_sweep.py`, `vw_demo.py` (FUERA del esquema classical)
- **Idea / diferencia:** para hacer tractable la enumeración VW, rankea y se queda con el top-L por un score barato. self_score(T)=(Σu_T)·∏(1−p_T); partner(T)=(Σu_T+u*)·∏(1−p_T)·p* usa el mejor V-pool global como surrogate de U, contando el presupuesto que T consume (lo que self_score ignora); variante entropy = self+λ·H(r_T). No elige UN pool: restringe el conjunto candidato para una búsqueda no-miope posterior.
- **Estado:** solo-notas (teoría/medición en la nota; código en augmented).
- **Hallazgos:** barridos de 6 regímenes: self_score gana en los 6 (mean L_min 0.7-12.7); partner domina a self en 4 de 6 y se mantiene ≤3 en mean L_min, escalando independiente de |S|. Adversarial: self_score L_min=8/15 (53%), partner=2/15 (13%). entropy NO ayuda. Resuelve que la enumeración VW es práctica en promedio; NO resuelve la cota de peor caso (max L_min=132 en n=15) ni si preserva la α-aproximación bajo lookahead.

### Selección de pool por solver (Mosek/Gurobi) para n grande — propuesta
- **Ubicación:** `docs/specs/2026-03-21-mosek-gurobi-pool-selection-design.md`
- **Idea / diferencia:** propone reemplazar la enumeración de `_myopic_best_pool` por un programa de optimización (cono exponencial Mosek o MILP con `addGenConstrLog` Gurobi) para escalar a n=30/50/100 donde la enumeración (hasta 79M pools) es inviable. Mismo score miope, solo que resuelto con solver. No cambia la política.
- **Estado:** solo-notas (estado 'Proposed').
- **Hallazgos:** ninguno empírico. Notable: es exactamente la técnica que classical YA usa (`solveConicSingle` con MOSEK); la propuesta es portarla al greedy enumerativo de augmented — y, de hecho, ya se materializó en `augmented/pool_solvers.py` (familia 5). Tabla de explosión combinatoria documentada (n=50,G=5 → 2.4M pools; n=100 → 79M).

> Las garantías teóricas heredadas (conic ≥1−1e−7; greedy estático (1−ε)/5; OrderedGreedy 1/e) vienen de Finster et al., no del código local. El código duplicado entre los dos solvers clásicos debería unificarse; la advertencia in-situ 'utility values erroneous from Gibbs sampling' y el bug de Gibbs afectan a la rama con conteos.

## Tabla-resumen

| algoritmo | tipo | estado | escala | en una frase |
|---|---|---|---|---|
| `_myopic_best_pool` | selector miope (baseline) | maduro | enumera C(n,G) | el tronco: elige el pool que maximiza P(r=0)·Σu, sin anticipar |
| `greedy_myopic_simulate` | simulador miope | maduro | n≤14 (ground truth 2^n) | corre la política miope sobre un perfil fijo |
| `greedy_myopic_expected_utility` | EU cerrada miope | maduro | n≤18 ramas exactas | EU analítica del miope con pesos de rama exactos |
| `_lookahead_best_pool` | greedy no-miope (1 paso) | experimental | pequeña | anticipa el primer paso, continúa miope; aproximado, no DP |
| `greedy_lookahead_simulate` | simulador no-miope híbrido | experimental | pequeña | lookahead solo en paso 0, miope después; sin integrar |
| `greedy_myopic_counting_*` | miope con conteo exacto | maduro | n≤~18 | posterior por enumeración de toda la historia; aquí vivió Fix #1 (+17%) |
| `greedy_myopic_gibbs_*` | miope con Gibbs MCMC | experimental | n~50+ (aprox.) | posterior aproximado por MCMC para escalar; mixing aún abierto |
| `vw_demo` (VW-A/VW-B) | super-nodos escalares | demo | conceptual | VW escalar all-clear = el paso miope exacto; no mejora el gap |
| `vw_restrict` / `_sweep` (partner) | poda top-L de super-nodos | experimental | n≤15 medido | partner poda W a ~1-3 candidatos, independiente de |S| |
| `exact_pool_pmf` | PMF posterior exacta | maduro | n≤8/14 (O(2^n)) | la conjunta verdadera que el miope aproxima por marginales |
| `gap_summary` / `aggregate` | métricas del independence gap | maduro | n=8 barrido | la brecha es 0 en la mediana con cola larga (TV hasta ~0.56) |
| `exact_greedy_myopic_expected_utility` | miope con scoring exacto | experimental | n≤8 | corrige el miope usando P(r=0|H) exacta; prueba de concepto |
| `infection_reward_greedy` (beta) | recompensa alternativa | maduro | n grande (shortlist) | miope + beta·info-gain; mueve la aguja solo en prevalencia moderada |
| `semi_utility` (alpha) | recompensa alternativa | experimental | n≤14 (recursión) | interpola limpieza dura↔blanda; placeholder, sin figuras ni tests |
| `hybrid_greedy_bruteforce` | híbrido greedy→DP | maduro | DP sobre ≤14 activos | greedy K pasos + DP exacto al final; puente miope↔óptimo |
| `mosek_best_pool` / `gurobi_best_pool` | exact-scoring 1 paso | maduro | n=30/50 G=5 | el argmax miope como cono/MILP en vez de enumeración |
| `_heuristic_best_pool` | fallback separable | maduro | cualquiera | top-G por u·(1−p); red de seguridad sin solver |
| `solve_optimal_dapts` (U_D_A) | DP óptimo augmented | maduro | n≤14 (real N=G=5,B=3) | el techo: horizonte completo con conteo exacto |
| `solve_classical_dynamic` (U_D) | DP óptimo binario | maduro | n≤14 | techo con test binario; U_D_A−U_D = beneficio del conteo |
| `solve_static_*` (U_s_NO/U_s_O) | baselines estáticos | maduro | n≤14 | pisos no-adaptativos de la cadena de desigualdades |
| `DaptsExactEnv` (PPO) | RL belief exacto | experimental | n=3 demo | MDP exacto para ver si PPO recupera el DP; sin métricas |
| `DaptsBucketEnv` (PPO) | RL + greedy híbrido | experimental | N=50 demo | RL elige el primer pool, el greedy juega el resto |
| `value_iteration` | DP como MDP (Bellman) | maduro | pequeña | el óptimo exacto escrito como value iteration |
| `tabular_q_learning` | Q-learning tabular | maduro | n≤3 validado | aprende Q* y converge al DP en instancias chicas |
| `AgentSelectionEnvB2..B5` (classical) | RL PPO anidado + MOSEK | abandonado | N=50 legacy | antecesor de DaptsBucketEnv; el resto lo jugaba un solver cónico |
| `solveConicGibbsGreedyDynamic` | greedy dinámico clásico (árbol) | experimental | N=50 (docs) | ancestro: conic + Gibbs + árbol binario completo |
| `solveConicGibbsGreedyDynamicCount` | greedy dinámico count-aware | experimental | N=50 (docs) | añade conteo exacto, una trayectoria; bug Gibbs declarado |
| `solveMILP` (classical) | MILP estático | experimental | N=50 | óptimo global aprox. estático; primo no-adaptativo del miope |
| `solveDynamic` (classical) | DP óptimo clásico | experimental | N≤10, real N=G=5,B=3 | el techo U_A^D; re-implementado en augmented/solver.py |
| `solveConicSingle` (MICOP) | selector miope cónico | experimental | O(N^5) | el miope resuelto óptimamente; corazón de la familia clásica |
| Garantía (1−1/e) Golovin-Krause | teoría | solo-notas | — | pregunta abierta Q4: ¿el conteo preserva submodularidad adaptativa? |
| Restricción VW por ranking | heurística (nota) | solo-notas | — | self_score/partner para podar super-nodos; código en augmented |
| Pool por solver para n grande | propuesta de diseño | solo-notas | — | portar solveConicSingle a augmented; ya materializado en pool_solvers.py |

## Cierre: qué está maduro, qué a medias, qué solo vive en notas

**Maduro y validado contra ground truth.** La rama miope completa —secuencial y conteo, en sus dos formas (simulación y EU cerrada)— está probada contra enumeración exacta (`_true_policy_eu`, gap < 1e-9), incluyendo el Fix #1 que corrigió una sobreestimación del +17% en `counting_expected_utility`. Todos los solvers (un paso y globales) tienen tests y están cableados en `comparison.py`/`cross_verification.py`/`overnight_experiments.py`. El beta-greedy y el híbrido greedy→DP tienen suites dedicadas que pasan y hallazgos numéricos registrados. El RL pedagógico (`value_iteration`, `tabular_q_learning`) está validado contra el DP.

**A medias.** El lookahead es la única ruptura real de la miopía en código, pero es parcial (anticipa solo el primer paso, continúa miope, pesos Poisson-Binomial) y apenas probado (un smoke test, sin integrar). La rama Gibbs sigue experimental por el fix de mixing aún abierto. La semi-utilidad corre pero no tiene tests propios, no aporta hallazgo cuantitativo y no aparece en las figuras finales del paper. El greedy de scoring exacto (`exact_greedy_myopic_expected_utility`) es prueba de concepto: falta el barrido comparativo que decida si la brecha cambia decisiones, y una versión que escale más allá de n≤8. El RL profundo (PPO) tiene modelos `.zip` de demostración pero ningún número de PPO vs. DP ni PPO vs. greedy guardado. En classical, el código duplicado entre los dos solvers debería unificarse y la rama count-aware arrastra el bug de Gibbs.

**Solo en notas.** La garantía (1−1/e) por submodularidad adaptativa (Golovin-Krause) sigue abierta: el hallazgo firme es negativo —el VW escalar = greedy miope (re-escritura, no mejora) y el gap crece con B (0.14→1.96)—, y la 'killer question' Q4 (si el conteo preserva submodularidad adaptativa) no está resuelta. La restricción VW por ranking vive como medición en la nota con código companion en augmented. La selección de pool por solver para n grande era una propuesta de diseño que ya se materializó en `pool_solvers.py`. Las garantías teóricas heredadas (conic ≥1−1e−7, greedy estático (1−ε)/5, OrderedGreedy 1/e) provienen de Finster et al., no del código local.