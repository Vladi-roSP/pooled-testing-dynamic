# Escalabilidad del greedy y estado del cálculo de la posterior

Dos preguntas: ¿el greedy corre a n grande (25, 50)?, y ¿en qué estado está el
cálculo de la probabilidad posterior (MCMC, Metropolis-Hastings, límites
probados)? Todo lo que sigue está verificado contra el código y medido en la M4.

## Parte 1 — ¿El greedy corre a n=25, 50?

Sí, dos de las tres variantes corren sin problema; la tercera no. Las variantes se
distinguen por cómo actualizan la creencia entre tests.

Mediciones (B=3, G=5, una instancia, perfil aleatorio):

| n | enumerar pools (G=5) | greedy secuencial (enum) | greedy secuencial (Mosek) | greedy Gibbs | greedy counting |
|---|---|---|---|---|---|
| 25 | 68 405 pools / 0.0 s | 0.1 s | 0.1 s | 0.1 s | **6.0 s** (enumera 2²⁵) |
| 50 | 2 369 935 pools / 0.4 s | 3.9 s | ~0 s (instantáneo) | 3.9 s | inviable (2⁵⁰) |

La lectura: el greedy **secuencial** y el **Gibbs** corren a n=50 en segundos. El
greedy **counting** ya cuesta 6 s a n=25 y es imposible a n=50. La selección de
pool por enumeración (C(n,≤G)) cuesta 0.4 s a n=50 con G=5, pero la versión con
solver Mosek la vuelve instantánea.

### Los dos cuellos de botella

El primero es la **actualización por conteo**: `bayesian_update_by_counting`
enumera los 2ⁿ perfiles, así que es O(2ⁿ·k). A n=25 son 33 millones de perfiles
(6 s por corrida); a n=50, 10¹⁵, imposible. Es el muro duro, y la razón de existir
del muestreo de Gibbs, que lo reemplaza.

El segundo es la **enumeración de pools**: `_myopic_best_pool` recorre
`all_pools_from_mask`, es decir C(activos, ≤G) pools por paso. A G=5 y n=50 son
2.4 millones (0.4 s, tolerable), pero crece como C(n,G): a G=10, n=50 serían 10¹⁰
pools, inviable. El selector Mosek/Gurobi (`pool_solvers.py`) resuelve la selección
como una optimización en vez de enumerar, y por eso es instantáneo a n=50; ya hay
tests de que la selección por solver corre en menos de 30 s a n=50.

### Qué variante usar a escala

Para n≳20, el greedy secuencial (independencia) o el greedy Gibbs (posterior por
muestreo). Si además G es grande, conviene el selector por solver para evitar la
enumeración de pools. El greedy counting solo sirve para n pequeño (≤~20), donde
es la referencia exacta.

## Parte 2 — Estado del cálculo de la posterior

La posterior P(X_i = 0 | Ax = r) se calcula por cuatro caminos en
`augmented/bayesian.py`, de exacto-pero-caro a aproximado-pero-escalable.

| método | función | qué calcula | costo | límite probado |
|---|---|---|---|---|
| secuencial Poisson-Binomial | `bayesian_update_single_test`, `bayesian_update` | aprox: trata las marginales como independientes (exacto solo si los pools son disjuntos) | O(k·G²) | cualquier n |
| conteo exacto | `bayesian_update_by_counting` | la posterior conjunta EXACTA | O(2ⁿ·k) | n≈20–25 (n=25 → 6 s) |
| PMF exacta de un pool | `exact_pool_pmf` | P(r=k \| historia) exacto | O(2ⁿ) | n≤18 (usado en utilidad esperada) |
| Gibbs (MCMC) | `gibbs_update` | exacto por componente, o MCMC si la componente es enorme | 2^\|componente\|, o MCMC | n=50 OK; validado vs exacto a n≤7 |

### ¿MCMC? ¿Metropolis-Hastings? Sí, ambos

El muestreo de Gibbs está implementado y reescrito esta sesión. No es un Gibbs de
sitio único ingenuo: descompone los agentes activos en componentes conexas
(`_connected_components`) y resuelve cada componente por separado. Si la componente
es chica (≤ `EXACT_ACTIVE_THRESHOLD` = 16 agentes) la enumera de forma exacta
(`_exact_component_marginals`); solo una componente más grande cae al MCMC. Ese
MCMC es un **Metropolis-Hastings** sobre **movimientos de camino alternante**
(`_propose_alternating_move`, `_alternating_move_component_marginals`): vectores del
núcleo de A con entradas en {−1, 0, +1} que equilibran cada test pero cambian el
conteo total, con aceptación por razón de priors. El estado inicial se siembra con
búsqueda de mínimos conflictos (`_find_valid_state`). En la práctica, como los
pools tienen tamaño ≤ G, las componentes activas casi siempre son chicas y el
camino exacto cubre todo; el MCMC es el respaldo para el caso raro de una
componente grande.

### Límites probados con cada algoritmo

El conteo exacto se usa como referencia hasta n≈14 en los tests, y aguanta hasta
n≈25 (6 s) antes de volverse impráctico. El Gibbs se validó contra el conteo
exacto en configuraciones n=5,6,7 (coinciden al 4.º decimal, porque ruteo a
exacto-por-componente), y su validez en la ruta MCMC se probó en 25 escenarios con
>7 agentes activos (`tests_gibbs_validity.py`), todos dentro de tolerancia; la
ergodicidad se probó forzando el MCMC sobre la instancia n=3 que antes lo rompía.
El greedy completo corre a n=50 con el secuencial o el Gibbs (3.9 s), según las
mediciones de arriba.

### Qué se ha intentado (historial del muestreador)

El muestreo tuvo cuatro intervenciones, útiles para no confundir versiones. La
primera fue un Gibbs de sitio único adaptado de Lopez et al. (commit 47e855a), no
ergódico. La segunda intentó arreglar la mezcla con movimientos de intercambio
(swap, 44f7e5f), insuficientes porque conservan el conteo total. La tercera fue un
parche provisional de validez (7ccad73, 2026-06-03): subió el umbral exacto, sembró
un estado consistente y añadió un guard para contar solo muestras válidas; cerró la
validez pero no la ergodicidad. La cuarta es la reescritura actual (547b324):
componentes conexas + exacto por componente + Metropolis de camino alternante. El
detalle está en `correcciones_gibbs.md`.

### Qué NO está implementado (solo en las notas)

La sección de inferencia aproximada de las notas menciona varios métodos que NO
están en el código y quedan como trabajo futuro: belief propagation (sum-product
sobre el grafo de factores), aproximaciones variacionales (mean-field), y
Sequential Monte Carlo / filtros de partículas para el escenario secuencial. Una
búsqueda en el repo no encuentra ninguno de estos; lo único de inferencia
aproximada implementado es el Gibbs.
