# Guía de lectura del paper (español)

Esta guía acompaña el borrador `dynamic_augmented_group_counting.tex`. Explica,
sección por sección, qué afirma el paper y por qué, para que el contenido quede
dominado antes de defenderlo o ampliarlo. El borrador está en inglés porque AAAI
lo exige; esta guía queda en español.

## El encuadre

El paper extiende el trabajo de testeo por grupos maximizador de bienestar de un
test binario a un test de conteo. En el esquema clásico un pool sobre un
subconjunto t responde solo si hay o no algún activo. El preprint binario
dinámico de Lopez, Marmolejo-Cossío, Tello Ayala y Parkes condiciona cada test en
el resultado conteo-no-cero o conteo-cero de los anteriores, y cierra señalando como
trabajo futuro los resultados de test más ricos que el binario. El esquema
augmented toma justo ese paso: el test revela el conteo exacto r = |t ∩ Z| de
activos en el pool. La motivación no es artificial. La counting devuelve un entero
(el cycle threshold) que es, a primer orden, monótono en la carga signal del pool,
de modo que el conteo es la idealización del dato cuantitativo que el laboratorio
ya produce. Ese encuadre define la novedad y, a la vez, la vía de colaboración:
no se compite con el preprint binario, se construye sobre él.

## El modelo y la jerarquía

La población tiene N individuos, cada uno con utilidad u_i y prior de estado latente
p_i, estados latentes independientes, presupuesto de B tests y pools de a lo más G. Se
gana u_i cuando i queda en algún pool con resultado r = 0. Una estrategia
dinámica F asigna el siguiente pool en función de la historia observada, y su
valor es la utilidad esperada sobre el perfil aleatorio Z.

El resultado de encuadre es la cadena

    U_single ≤ U_s_NO ≤ U_s_O ≤ U_D ≤ U_D_A ≤ U_max,

que ordena el óptimo bajo regímenes cada vez más potentes: testeo individual,
estático no solapado, estático solapado, dinámico binario, dinámico augmented y
la cota de información total U_max = Σ u_i q_i. El eslabón nuevo es U_D ≤ U_D_A:
el conteo domina al binario porque su resultado refina al binario. El tamaño de
ese hueco es lo que el paper mide. El ejemplo con tests {0,1}=1 y {1,2}=0
registro la deducción cruzada: de r_2 = 0 se sabe que 1 y 2 están limpios, y con
r_1 = 1 se concluye que 0 está activo, sin haberlo testeado solo.

## El resultado teórico fuerte: dureza #P

La inferencia posterior bajo conteo es el aporte teórico central. La posterior
sobre perfiles es proporcional a la indicadora de Ax = r por el prior, y su
constante de normalización Z cuenta soluciones 0/1 de un sistema de restricciones
de cardinalidad. La Proposición 1 consulta que calcular la marginal exacta
P(X_i = 0 | Ax = r) es #P-hard, por reducción de #Exact Cover: cada conjunto del
sistema es una variable, cada elemento del universo es un test con conteo
observado 1, y con priors uniformes Z cuenta exactamente los exact covers. La
lectura es que las mismas deducciones que hacen útil al conteo vuelven intratable
la inferencia exacta en general. El paper acota esa dureza identificando los
regímenes que sí son tratables: pools disjuntos (la posterior factoriza), pools
laminares o anidados (programación dinámica sobre el árbol de inclusión),
treewidth acotado (junction tree) y pocos tests (exponencial en k, polinomial en
N).

## La sutileza de ergodicidad en el muestreo

Fuera de esos regímenes se estiman marginales muestreando perfiles consistentes
con Ax = r. Un paso de sitio único casi nunca preserva la factibilidad, porque un
agente dentro de un pool ajustado no puede cambiar solo. Los swaps reparan eso
dentro de un pool pero conservan el conteo total de activos, así que la cadena
queda atrapada en el nivel de conteo donde arrancó: en {0,1}=1, {1,2}=1 con prior
0.15 la posterior exacta es (0.15, 0.85, 0.15), pero un generator de solo swaps
devuelve (0,1,0) o (1,0,1) según la semilla. La solución del paper tiene dos
partes. Primero descompone los agentes activos en componentes conexas, ligadas
por tests compartidos; la posterior factoriza entre componentes y cada una se
resuelve por separado y de forma exacta por enumeración cuando es pequeña, lo que
cubre toda escala real porque el límite aplica por componente. Para una
componente demasiado grande corre un Metropolis sobre movimientos de camino
alternante: vectores del núcleo de A con entradas en {−1,0,+1} que equilibran
cada test pero cambian el conteo total, como (+1,−1,+1) en el ejemplo, que es el
movimiento que a los generators locales les falta.

## Los algoritmos y dos correcciones

El óptimo se calcula con un programa dinámico sobre el estado (paso, perfiles aún
consistentes, conjunto limpiado); el test augmented ramifica en |t|+1 resultados
contra los dos del binario, y solo es factible para N ≤ 14. El greedy miope elige
el pool que maximiza ∏(1−p̃_i)·Σ u_i, y una observación útil es que la elección
miope coincide entre binario y augmented, porque solo r = 0 da utilidad
inmediata; el conteo ayuda únicamente en los posteriores de pasos futuros.

Dos correcciones del código importan para que la evaluación sea fiel. La primera
es la cosecha de limpios deducidos: la utilidad se acredita solo al colocar a un
individuo en un pool con r = 0, así que un individuo deducido limpio conserva su u_i
y solo se cobra testeándolo en un pool de r = 0 garantizado; filtrarlo de los
pools futuros tira esa utilidad, y mantenerlo elegible recupera el hueco (en una
instancia, el greedy de conteo sube de 46.0 a 57.6 frente al óptimo 57.8). La
segunda es el peso de las ramas de resultado: pesarlas con una Poisson-Binomial
de las marginales asume una independencia que el condicionar en la historia ya
destruyó, y puede equivocar el valor de una política por dos dígitos; las ramas
deben pesarse con la P(r | historia) exacta sobre perfiles consistentes, con lo
que el valor cerrado iguala al de simular la política.

## Los experimentos

Con p_i ~ U(0,1) y u_i uniforme en {1,2,3}, sobre 200 instancias por
configuración y con el código corregido, la cadena se cumple en todas las
instancias (40 en N=7, más pesado) y el beneficio augmentado U_D_A − U_D crece
con la escala: +0.63% en N=3, +3.97% en N=5 y +5.07% en N=7. El régimen de
presupuesto bajo registro el hueco más chico porque solo se condiciona en un
resultado. La tendencia respalda el mecanismo: el conteo paga vía mejores
posteriores sobre un horizonte más largo, así que su valor sube con el
presupuesto y la población.

## Qué queda abierto

Quedan dos preguntas que el paper deja planteadas. La teórica es si el valor por
paso bajo conteo es adaptativo-submodular, lo que daría al greedy una garantía
tipo (1 − 1/e). La empírica es cómo se comporta la ventaja sobre datos reales de
counting, donde el conteo se observa solo a través de un cycle threshold ruidoso.
Ambas deciden si el testeo augmented puede mejorar el cribado real de salud
pública, que es el eje que valora el track AISI.
