# Estado del proyecto — Dynamic Augmented Pooled Testing (lenguaje sencillo)

_2026-05-25. Objetivo de este documento: **reconstruir el entendimiento** del
proyecto, que se avanzó rápido con LLMs durante el semestre. Está escrito en
lenguaje simple. Revisado contra el código real (con ayuda de Codex)._

---

## 0. Mini-glosario (para no perderse)

- **DP (programación dinámica):** resolver el problema **exacto** probando todas
  las decisiones futuras posibles. Da el óptimo pero es lento.
- **warm-up:** la primera fase del paper — resolver casos **chicos** a fuerza bruta.
- **greedy (avaricioso):** estrategia simple que en cada paso elige lo que se ve
  mejor **ahora**, sin planear el futuro.
- **posterior:** la probabilidad **actualizada** de que alguien esté infectado,
  después de ver resultados de tests.
- **posterior conjunto exacto:** la probabilidad real de que **un grupo** salga
  positivo. Ojo: tras varios tests, las personas quedan **relacionadas**, así que
  no basta con multiplicar sus probabilidades individuales.
- **info cruzada:** cuando dos tests comparten personas, el resultado de uno da
  pistas sobre el otro.
- **super-nodos (VW):** tratar a un grupo de personas ya tocadas como **un solo
  paquete** (idea de Francisco).
- **PMF (del conteo):** la lista de probabilidades de cada posible **número** de
  positivos en un grupo (0, 1, 2, …).
- **bucket (cubeta):** agrupar personas en categorías (por salud y utilidad) para
  que el estado **no crezca** con N. Sirve para escalar.
- **Gibbs / mixing:** un método de muestreo aleatorio para estimar
  probabilidades difíciles; "mezcla mal" = no recorre bien los escenarios y
  sesga el resultado.

---

## 1. ¿De qué va el proyecto? (en una frase)

Tienes `n` personas, cada una con una probabilidad de estar infectada `p_i` y un
valor `u_i` por declararla sana. Con pocos tests `B` (cada uno mezcla hasta `G`
personas) quieres **maximizar el valor de la gente que logras probar sana**. La
versión "augmented" supone que el test te dice el **número exacto** de infectados
en el grupo (no solo sí/no).

---

## 2. Qué YA está hecho (y funciona)

| Idea del paper | Dónde está en el código | Estado |
|---|---|---|
| Modelo formal + cómo medir una estrategia | `core.py`, `strategy.py`, `simulator.py`, `expected_utility.py` | ✅ |
| **Solver óptimo exacto (DP)** del warm-up | `solver.py` (`solve_optimal_dapts`) | ✅ (solo n≤14) |
| Actualizaciones de probabilidad (Bayes) | `bayesian.py`: por un test, por **historia completa** (info cruzada), y por **Gibbs** | ✅ |
| Greedy (varias versiones) | `greedy.py`: miope, lookahead **de primer paso**, por conteo, gibbs | ✅ |
| Variante con peso β / ganancia de info | `infection_reward_greedy.py` | ✅ |
| Mezcla greedy + DP exacto en lo que sobra | `hybrid_solver.py` (cae a greedy si quedan >14 activos) | ✅ |
| Quitar de los grupos a quien ya tiene status conocido | `core.py` (`compute_active_mask`) + greedy | ✅ |
| Sacar y dibujar el **árbol de decisión** | `tree_extractor.py`, `tree_visualizer.py` | ✅ |
| Comparar estrategias (la cadena U_single ≤ … ≤ U_max) | `baselines.py`, `static_solver.py`, `classical_solver.py`, `comparison.py` | ✅ |
| **Medir el error de "multiplicar probabilidades"** (la duda del §"Quick note") | `independence_gap.py` (incluye un greedy **exacto** para n chico) | ✅ medido |
| Gibbs + meta-parámetro β (régimen de prevalencia, +26%) | `bayesian.py`, `gibbs_analysis.py`, `phase3_findings.tex` | ✅ |
| **Estudio preliminar de prevalencia** (augmented vs clásico al subir infección) | `experiments.py` (barre rangos low/med/high) | ✅ preliminar |
| Exploración de **utilidad nueva** (interpolada por α) | `semi_utility.py` | ✅ (no cubre falsos pos/neg) |
| **Variant of Greedy / VW** (super-nodos) | `vw_demo.py`, `vw_restrict.py`, `vw_restrict_sweep.py`, notebooks 06/07 | ✅ ver abajo |
| RL pedagógico | `rl_examples.py`: value iteration (**= DP**) y Q-learning (se **acerca** al óptimo) | ✅ |
| **RL adaptado a augmented (PPO)** | `rl_env.py` (entorno exacto + bucket), `rl_train.py` | ✅ corre; ver §4 |

**Lo que aprendimos del VW:** la versión "all-clear" **da exactamente lo mismo**
que el greedy miope (es una re-escritura, no algo nuevo todavía). El paquete
descrito solo con números (peso, prob, utilidad) **no ve** la PMF del conteo, así
que **no cierra** la distancia con el óptimo, que **crece con B**. El heurístico
`partner` **reduce mucho** la búsqueda entre los super-nodos, pero **no garantiza
top-2**: en los barridos el peor caso (`max L_min`) llega a 14.

---

## 3. Qué quedó a medias o sin empezar

1. **Test realista (modelo φ/π).** Hoy todo asume el caso **idealizado**: el test
   te da el conteo verdadero, sin error. El modelo realista (con ruido, tipo
   "bolas en urnas") **no está hecho** — el propio paper dice "no lo haremos aún".

2. **La probabilidad conjunta exacta.** Ya **medimos** que "multiplicar
   probabilidades individuales" ≠ "probabilidad real del grupo"
   (`independence_gap.py`), e incluso hay un greedy exacto para n chico. Pero el
   greedy "normal" **sigue usando la multiplicación** (la aproximación). Falta
   decidir si eso importa en la práctica y/o usar el cálculo exacto de forma que
   escale.

3. **Teoría del VW (lo de Francisco).** Sigue **abierto**: ¿el VW da una
   **garantía** matemática tipo (1 − 1/e)? Y faltan aclarar 3 cosas de su
   formulación: ¿se permite un paquete de **una sola** persona? ¿un paquete cuenta
   como **1 lugar** o como `|A|` lugares del límite G? ¿pueden ir **dos paquetes
   que se solapan** en el mismo test? (ver `docs/notes/vw_submodularity.md`).

4. **Peores casos (ratios).** Solo tenemos comparaciones promedio (una tabla
   n=5). Faltan **cotas de peor caso** entre las estrategias en el modo augmented.

5. **Robustez a alta prevalencia.** Hay un estudio **preliminar**
   (`experiments.py`), pero falta **consolidarlo y validarlo** como resultado
   final (la hipótesis: en augmented "no se rompe tanto" al subir la infección).

6. **Direcciones grandes del paper.** **Fairness** e **incentivos**: sin empezar.
   **Falsos positivos/negativos** y **4 utilidades por persona**: sin empezar
   (lo único parecido es `semi_utility.py`).

7. **RL a escala.** Falta **entrenar de verdad** el PPO bucket (N=50, ~200k pasos)
   y reportar "RL le gana al greedy". Lo de hoy fueron pruebas cortas.

---

## 4. Limitaciones técnicas (por qué no es tan fácil)

- **El DP exacto solo aguanta n≤14.** El número de escenarios posibles crece como
  2^n y explota (lo viste en el notebook 07, Fig 4: la "pared" cae en n≈14).
- **El Bayes exacto también explota** (enumera todos los mundos posibles). Gibbs
  es la alternativa **aproximada**. ⚠️ Corrección 2026-06-03: el bug de "mixing"
  NO estaba realmente arreglado — el muestreador **contaba perfiles inválidos** en
  el camino MCMC (>7 activos). Ya se aplicó un fix (exacto sobre el conjunto activo
  ≤16 + guard de validez). Ver `gibbs_validez_2026-06-03.md`.
- **"Multiplicar probabilidades" sesga.** Por eso el greedy a veces **sobrestima**
  su propio desempeño (asume que las personas son independientes tras los tests,
  y no lo son). Es la limitación conceptual más importante.
- **Algunos solvers necesitan licencia** (MOSEK/Gurobi en `pool_solvers.py` y en
  varios experimentos) → no corren sin licencia. El RL nuevo **evita esto** usando
  el greedy de augmented.
- **El RL tiene dos modos con sus límites:** el **exacto** ve todos los 2^n
  escenarios → solo n chico (misma pared que el DP). El **bucket** escala, pero es
  aproximado; además sus cubetas de utilidad están pensadas para valores {1,2,3} y
  **se colapsan** con los CSV (utilidades entre 0 y 1).
- **Resultados PPO aún no "oficiales":** ya hay una ruta para comparar PPO vs DP y
  en una prueba corta (CSV N=3) dio igual al óptimo, pero **falta guardarlo como
  resultado reproducible** (un test/medición versionada).

---

## 5. Próximos pasos sugeridos (en orden)

1. **Cerrar la duda de la probabilidad conjunta** (punto 2): comparar bien
   "multiplicar" vs "exacto" y ver si un greedy exacto mejora en n chico. Conecta
   directo con la teoría del VW.
2. **Atacar la garantía del VW** y **aclarar con Francisco** las 3 dudas de su
   formulación.
3. **Entrenar el RL a escala** (PPO bucket N=50) y reportar vs greedy.
4. **Consolidar el estudio de prevalencia** (augmented vs clásico) como resultado.
5. (Largo plazo) Modelo de test realista y/o utilidad con falsos pos/neg.
