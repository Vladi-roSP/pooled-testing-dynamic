# Correcciones al código de Gibbs

Sí, se corrigió el muestreador de Gibbs. Esta sesión lo reescribió para volverlo
ergódico, y esa reescritura reemplazó dos parches anteriores que solo atacaban el
síntoma. El código vive en `augmented/bayesian.py` y el commit es
`547b324` (fix: ergodic Gibbs via connected components + alternating-path Markov
moves).

## Qué hace el Gibbs y por qué se necesita

`gibbs_update` aproxima por muestreo las probabilidades posteriores de estado latente
cuando enumerar los 2^n perfiles es inviable. La inferencia exacta es intratable en
general, así que para escalas grandes se muestrean perfiles consistentes con los
conteos observados (la restricción A x = r) y se estiman las marginales como
frecuencias. La dificultad de fondo es diseñar movimientos de la cadena que
preserven todos los conteos observados y, aun así, recorran todo el espacio
factible.

## El bug: la cadena no era ergódica

El muestreador anterior quedaba atrapado. Sus movimientos —redrawar un sitio,
intercambiar un activo por un limpio dentro de un pool, y bloques— preservan todos
el número total de activos, así que la cadena nunca podía cruzar entre
configuraciones con distinto total y devolvía marginales sesgadas en pools
solapados.

El caso mínimo lo deja claro. Con tests {0,1}=1 y {1,2}=1 y prior 0.15 para los
tres, los perfiles factibles son (1,0,1) y (0,1,0), que difieren en las tres
coordenadas y en el total de activos (dos contra uno). La posterior exacta es
[0.15, 0.85, 0.15], pero el muestreador viejo devolvía [0,1,0] o [1,0,1] según la
semilla: confiado y falso. La causa exacta es que para saltar de un perfil al otro
hay que voltear los tres agentes a la vez, y ningún movimiento local lo permite.

## La corrección, en dos partes

La reescritura descompone primero a los individuos activos en componentes conexas,
donde dos individuos están conectados si comparten un test (`_connected_components`,
`_component_tests`). Como ningún test cruza componentes, la posterior factoriza
entre ellas y cada una se resuelve por separado. Si la componente es pequeña se
enumera de forma exacta (`_exact_component_marginals`); como el límite aplica por
componente y no sobre todos los activos a la vez, esto cubre cualquier escala real
del proyecto y, de hecho, elimina el bug en todos los casos prácticos, porque el
caso mínimo es una sola componente de tres agentes que se resuelve exacta.

La segunda parte cubre el caso raro de una componente demasiado grande para
enumerar. Ahí corre un Metropolis con movimientos de camino alternante
(`_propose_alternating_move`, `_alternating_move_component_marginals`): vectores con
entradas en {−1, 0, +1} que equilibran cada test pero pueden cambiar el conteo
total, como (+1, −1, +1), que es justamente el salto entre (0,1,0) y (1,0,1). Ese
es el movimiento que a la versión anterior le faltaba, y es lo que restaura la
ergodicidad.

## Lo que se eliminó

La reescritura dejó obsoletas dos piezas del parche anterior, `_exact_active_marginals`
(la enumeración monolítica sobre todos los activos) y la constante
`EXACT_ACTIVE_FALLBACK_CAP`, que se borraron. La constante `EXACT_ACTIVE_THRESHOLD`
se conservó, pero ahora es el límite por componente, no sobre el conjunto activo
completo.

## Parches anteriores que esto reemplazó

El muestreador tuvo intervenciones previas que conviene conocer para no
confundirlas con esta. El commit `7ccad73` (3 de junio de 2026) fue un parche
provisional de validez: subió el atajo exacto a un umbral de agentes activos,
sembró un estado inicial consistente y añadió un guard para contar solo registros
válidas. Cerró un problema de consistencia, pero no tocó la raíz de la mezcla, así
que la cadena seguía sin ser ergódica. Antes de él, `44f7e5f` ya había intentado
arreglar la mezcla con movimientos de swap, insuficientes por la misma razón. La
reescritura de esta sesión es la que ataca la causa: la falta de movimientos que
cambien el conteo total.

## Cómo se validó

Forzando el camino del MCMC (bajando el umbral por componente) sobre el caso
mínimo, las marginales convergen a [0.15, 0.85, 0.15] en todas las semillas
probadas, con error máximo por debajo de 0.03; antes el error era de 0.15 y
dependía de la semilla. Las consultas viven en `tests_correctness_fixes.py` (la de
ergodicidad y la de componentes independientes) y en `tests_gibbs_validity.py`
(25 escenarios, todos dentro de tolerancia). La suite completa quedó en verde.

## La corrección de julio: el equilibrio detallado

La auditoría del 6 de julio de 2026 encontró que la reescritura de junio dejó un
segundo defecto, independiente del primero. Los movimientos de camino alternante
sí hacen irreducible a la cadena (se verificó por enumeración exacta en cinco
topologías), pero la propuesta es asimétrica: la probabilidad de proponer el
camino de ida depende del número de parejas elegibles en cada paso de reparación,
y ese número difiere entre el estado actual y el propuesto. La aceptación usaba
solo el cociente de priors (Metropolis puro), que exige propuesta simétrica, así
que la cadena convergía con toda confianza a una distribución estacionaria
equivocada. En el contraejemplo mínimo (tests {0,1,2}=1 y {2,3,4}=1 con priors
heterogéneos) la distancia de variación total entre la estacionaria de la cadena
y la posterior exacta era 0.067, con 6.7 puntos porcentuales de error en una
marginal, estable en todas las semillas.

El defecto era invisible para la suite porque el único test que ejercitaba el
MCMC usaba una fibra de dos estados donde toda propuesta exitosa es determinista
y la asimetría desaparece. Los resultados con n≤14 nunca tocaron esta rama (el
umbral exacto por componente es 16), así que ningún número publicado cambia.

La corrección es el factor de Hastings por camino espejo: al construir la
propuesta se registra la secuencia de reparaciones, y la probabilidad del camino
inverso se computa releyendo esa misma secuencia desde el estado propuesto, con
los roles de ida y vuelta invertidos. La aceptación pasa a ser
min(1, π(z')·q(rev)/π(z)·q(fwd)). Verificación: la matriz de transición exacta
de la cadena corregida (enumerando todas las ramas del generador aleatorio) da
distancia de variación total 0.000000 contra la posterior en las cinco
topologías, y el error end-to-end del muestreador cae de 0.067 a ruido de Monte
Carlo (~0.003). El test de regresión con el contraejemplo vive en
`tests_correctness_fixes.py`.

## Resumen

El muestreador de Gibbs necesitó dos correcciones. La de junio atacó la
irreducibilidad: la descomposición en componentes con resolución exacta por
componente y los movimientos de camino alternante que cruzan niveles de conteo.
La de julio atacó el equilibrio detallado: el factor de Hastings por camino
espejo que la propuesta asimétrica exigía. Con ambas, la cadena es irreducible y
converge a la posterior correcta, verificado por matriz de transición exacta y
por comparación directa con la enumeración. La versión navegable de esta
explicación está en la nota del vault sobre la no ergodicidad de Gibbs.
