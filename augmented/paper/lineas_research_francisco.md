# Líneas de research para Francisco: qué tan bueno es el greedy y cómo certificarlo

Tres resultados preliminares, todos sobre la pregunta central: el greedy escala
pero no podemos calcular el óptimo a gran escala, así que ¿cómo sabemos qué tan
bueno es y cómo lo mejoramos? Setup común: prior $p_i\sim U(0,1)$,
$u_i\sim\{1,2,3\}$, DP exacto como referencia donde es computable.

## 1. De qué está hecho el hueco del greedy (descomposición)

El hueco de optimalidad del greedy miope (~5-6%) se separa en dos causas
independientes: la **miopía** (elegir por recompensa inmediata) y la
**aproximación de independencia en la selección** (puntuar con el producto de
marginales en vez de la conjunta exacta). Comparando el greedy estándar contra
`exact_greedy_myopic_expected_utility` (que puntúa con $P(r{=}0\mid H)$ exacta)
contra el óptimo:

| n | hueco total | miopía pura | costo de la independencia |
|---|---|---|---|
| 5 | 5.26% | 4.24% | 1.02 pp |
| 6 | 6.13% | 4.49% | 1.64 pp |
| 7 | 5.89% | 4.44% | 1.46 pp |

Alrededor de tres cuartas partes del hueco es miopía intrínseca —la tendrías aun
con un scoring perfecto— y el cuarto restante es la independencia, que crece con n.
La implicación es operativa: la palanca grande es atacar la miopía (lookahead), y
la palanca chica y barata es el scoring exacto, que ya está implementado.

## 2. La recuperación del lookahead colapsa con el horizonte

El hueco del greedy es un fenómeno de primera jugada, así que un lookahead que
anticipe el primer paso debería recuperarlo. La pregunta es cuánto, y cómo depende
del horizonte $B$. Midiendo el lookahead de un paso contra el miope y el óptimo
(n=6, G=4):

| B | hueco miope | hueco lookahead | recupera |
|---|---|---|---|
| 1 | 0% | 0% | 100% |
| 2 | 2.65% | 0.02% | 99% |
| 3 | 4.90% | 2.93% | 40% |
| 4 | 6.99% | 5.84% | 16% |

El patrón es nítido y monótono: con horizonte corto el lookahead de un paso lo
cierra casi todo (a B=2 iguala al óptimo, porque anticipar el único paso futuro es
la optimización completa), pero conforme el horizonte crece recupera cada vez menos
—40% a B=3, 16% a B=4— porque deja más pasos futuros jugados de forma miope. La
lectura es una ley de diseño: la profundidad de anticipación necesaria escala con
el horizonte; un lookahead de profundidad fija no basta para horizontes largos. Es
el complemento dinámico del descubrimiento del horizonte (el beneficio del conteo y
el costo de la miopía crecen ambos con $B$).

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

El resultado más fuerte y nuevo es la ley de recuperación del lookahead (sección 2):
sale de nuestro propio hallazgo del horizonte, es limpio y sugiere una pregunta
teórica concreta —¿cuánto hueco cierra un lookahead de profundidad $d$ como función
de $B$, y existe un $d(B)$ que garantice estar dentro de $\epsilon$ del óptimo? La
descomposición (sección 1) es el soporte que dice dónde está el hueco. Y la cota
(sección 3) es el problema abierto de certificación: el prototipo de hindsight está
hecho y registro por qué necesitamos la versión con penalización, que es lo que
convertiría todo esto en un resultado de certificación a escala.
