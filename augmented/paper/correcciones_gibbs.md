# Correcciones al código de Gibbs

Sí, se corrigió el muestreador de Gibbs. Esta sesión lo reescribió para volverlo
ergódico, y esa reescritura reemplazó dos parches anteriores que solo atacaban el
síntoma. El código vive en `augmented/bayesian.py` y el commit es
`547b324` (fix: ergodic Gibbs via connected components + alternating-path Markov
moves).

## Qué hace el Gibbs y por qué se necesita

`gibbs_update` aproxima por muestreo las probabilidades posteriores de infección
cuando enumerar los 2^n perfiles es inviable. La inferencia exacta es intratable en
general, así que para escalas grandes se muestrean perfiles consistentes con los
conteos observados (la restricción A x = r) y se estiman las marginales como
frecuencias. La dificultad de fondo es diseñar movimientos de la cadena que
preserven todos los conteos observados y, aun así, recorran todo el espacio
factible.

## El bug: la cadena no era ergódica

El muestreador anterior quedaba atrapado. Sus movimientos —resamplear un sitio,
intercambiar un infectado por un sano dentro de un pool, y bloques— preservan todos
el número total de infectados, así que la cadena nunca podía cruzar entre
configuraciones con distinto total y devolvía marginales sesgadas en pools
solapados.

El caso mínimo lo deja claro. Con tests {0,1}=1 y {1,2}=1 y prior 0.15 para los
tres, los perfiles factibles son (1,0,1) y (0,1,0), que difieren en las tres
coordenadas y en el total de infectados (dos contra uno). La posterior exacta es
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
sembró un estado inicial consistente y añadió un guard para contar solo muestras
válidas. Cerró un problema de consistencia, pero no tocó la raíz de la mezcla, así
que la cadena seguía sin ser ergódica. Antes de él, `44f7e5f` ya había intentado
arreglar la mezcla con movimientos de swap, insuficientes por la misma razón. La
reescritura de esta sesión es la que ataca la causa: la falta de movimientos que
cambien el conteo total.

## Cómo se validó

Forzando el camino del MCMC (bajando el umbral por componente) sobre el caso
mínimo, las marginales convergen a [0.15, 0.85, 0.15] en todas las semillas
probadas, con error máximo por debajo de 0.03; antes el error era de 0.15 y
dependía de la semilla. Las pruebas viven en `tests_correctness_fixes.py` (la de
ergodicidad y la de componentes independientes) y en `tests_gibbs_validity.py`
(25 escenarios, todos dentro de tolerancia). La suite completa quedó en verde.

## Resumen

El muestreador de Gibbs pasó de dar respuestas sesgadas y dependientes de la
semilla en pools solapados a converger a la posterior correcta, gracias a la
descomposición en componentes con resolución exacta por componente y a los
movimientos de camino alternante para las componentes grandes. La versión
navegable de esta explicación está en la nota del vault sobre la no ergodicidad de
Gibbs.
