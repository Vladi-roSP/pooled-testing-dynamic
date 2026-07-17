# Líneas de research para Francisco: qué tan bueno es el greedy y cómo certificarlo

Tres resultados preliminares, todos sobre la pregunta central: el greedy escala
pero no podemos calcular el óptimo a gran escala, así que ¿cómo sabemos qué tan
bueno es y cómo lo mejoramos? Setup común: prior $p_i\sim U(0,1)$,
$u_i\sim\{1,2,3\}$, DP exacto como referencia donde es computable.

## 1. De qué está hecho el hueco del greedy (descomposición en tres peldaños)

El hueco de optimalidad del greedy miope (~5-7%) se separa en TRES causas, no
dos. La tabla vieja fundía en un solo "costo de la independencia" dos cosas
distintas: puntuar con producto en vez de conjunta (**scoring**) y acarrear
marginales secuenciales en vez de exactas (**propagación**). El peldaño
intermedio del propio repo (`greedy_myopic_counting_expected_utility`:
marginales exactas, scoring por producto) las separa. Con 30 instancias
sembradas por n (misma receta, B=G=3):

| n | hueco total | miopía pura | propagación | scoring puro |
|---|---|---|---|---|
| 5 | 5.37% | 3.90% | 1.45 pp | 0.02 pp |
| 6 | 5.32% | 4.30% | 0.84 pp | 0.18 pp |
| 7 | 6.98% | 5.36% | 1.48 pp | 0.14 pp |

Alrededor de tres cuartas partes del hueco sigue siendo miopía intrínseca, pero
la lectura operativa del cuarto restante cambió: dentro del viejo "costo de la
independencia", la **propagación domina** (99% en n=5, 80% en n=6, 92% en n=7)
y el scoring conjunto puro es casi gratis de ignorar. La palanca grande sigue
siendo la miopía (lookahead); la palanca chica y barata es **propagar
marginales exactas** (deducciones + counting/gibbs, ya implementado), no el
scoring conjunto. Nota de honestidad: la escalera greedy ≤ counting ≤ exacto
no es teorema por instancia (1 violación exacto&lt;counting en 90 instancias);
la atribución es agregada.

## 2. La recuperación del lookahead NO colapsa con el horizonte (errata)

> **Errata (17-jul-2026).** La versión anterior de esta sección reportaba la
> ley 99/40/16 y concluía "la profundidad de anticipación necesaria escala con
> el horizonte". Esa medición usaba el lookahead con cableado legacy (updates
> secuenciales + pesos Poisson-Binomial): **medía la degradación del cableado,
> no la miopía**. Re-medido con ambos cableados sobre instancias idénticas, el
> colapso desaparece.

Sobre 30 instancias sembradas (n=6, G=4; `experiments_lookahead_exact.py`,
CSV en `data/lookahead_law_rewired.csv`), lookahead de un paso re-planificado
en cada jugada, dos cableados — legacy (selección con PB + secuencial, valor
evaluado exacto) y exacto (selección y pesos sobre perfiles consistentes):

| B | hueco miope | recupera (legacy) | recupera (exacto) |
|---|---|---|---|
| 1 | 0% | 100% | 100% |
| 2 | 1.56% | 92% | 100% |
| 3 | 7.78% | 42% | 89% |
| 4 | 11.66% | 38% | 93% |

El cableado legacy reproduce el colapso publicado (92→42→38, la forma de
99/40/16); el exacto recupera ~90% del hueco miope a TODO horizonte medido. La
lectura corregida: **la anticipación de un paso, bien cableada, basta en este
rango de horizontes**; lo que se degradaba con B era el error de inferencia
componiéndose con la profundidad. La pregunta teórica interesante ya no es
"¿qué profundidad d(B) necesito?" sino "¿cuánta *calidad de inferencia*
necesita el lookahead para no perder su valor?" — y el residuo de ~10% que el
exacto no recupera en B≥3 es la miopía de segundo orden real, mucho más chica
de lo que creíamos.

## 3. Certificar el greedy a escala: la cota superior por información perfecta

Como el óptimo es incomputable a n grande, la única forma de certificar el greedy
es acotarlo por arriba. La cota más simple es la de información perfecta
(hindsight): si conocieras el perfil $Z$, limpiarías a las $B\cdot G$ personas
limpias de mayor utilidad, y su valor esperado acota por arriba al óptimo dinámico.

| n | U_DA (óptimo) | U_PI (cota) | U_DA/U_PI | (U_PI − greedy)/U_PI |
|---|---|---|---|---|
| 4 | 2.7250 | 2.8415 | 0.959 | 9.28% |
| 5 | 4.4196 | 4.7938 | 0.922 | 13.93% |
| 6 | 5.5190 | 6.2493 | 0.883 | 16.23% |
| 7 | 5.4667 | 6.5343 | 0.837 | 21.64% |

La cota es válida (U_PI ≥ U_DA en todos los n) y computable a cualquier escala: a
n=50, B=G=5 da U_PI=46.9 contra greedy=19.5, es decir certifica que el greedy está
a lo más a 58% del óptimo, sin haber calculado el óptimo. El problema es que la
cota se afloja con n (U_DA/U_PI baja de 0.96 a 0.84) y a escala el certificado es
débil. La razón es estructural: cuando el presupuesto $B\cdot G$ se acerca al número
de limpios, la información perfecta limpia a casi todos y la cota tiende a $U^{\max}$,
ignorando lo difícil que es deducir sin conocer $Z$.

El prototipo funciona y deja clara la pieza que falta: una cota por **relajación de
información con penalización** (Brown–Smith–Sun), que cobra por usar información
futura y aprieta el certificado. Esa penalización es el contenido teórico
publicable, y es terreno natural de discusión con Francisco.

## Qué llevar a la reunión

El resultado más fuerte cambió de signo y eso es lo que hay que contar: la
"ley de recuperación del lookahead" era un artefacto de cableado (sección 2,
errata), y la versión corregida es igual de interesante — lookahead de un paso
bien cableado recupera ~90% del hueco a todo horizonte medido, así que la
pregunta teórica se desplaza de la profundidad $d(B)$ a la interacción
lookahead × calidad de inferencia. La descomposición en tres peldaños
(sección 1) apunta la palanca barata a la propagación de marginales exactas.
Y la cota (sección 3) sigue siendo el problema abierto de certificación: el
prototipo de hindsight está hecho y registro por qué necesitamos la versión
con penalización, que es lo que convertiría todo esto en un resultado de
certificación a escala.
