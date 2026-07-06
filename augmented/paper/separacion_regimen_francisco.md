# La separación aumentado vs clásico: dónde aparece, con greedy

Experimento dirigido a la pregunta que organiza el proyecto de Francisco: cuánto
vale el régimen aumentado (la consulta revela el conteo exacto $R$) sobre el clásico
(solo conteo-no-cero/conteo-cero), y bajo qué condiciones aparece. Se hace empíricamente y
con greedy, que es el método viable a escala, y se consultan sus dos hipótesis de
régimen: prevalencia intermedia y tamaño de pool $G>3$.

## Método y corrección

Se compara el greedy del régimen aumentado contra el greedy del régimen clásico.
Ambos usan exactamente la misma regla de selección miope; lo único que difiere es
la actualización de creencias: el aumentado condiciona en el conteo $r$, el clásico
solo en el signo $r>0$ frente a $r=0$. Así, la diferencia de bienestar aísla el
valor de conocer $R$ bajo la misma heurística. La maquinaria binaria se validó antes
de medir nada: su actualización de un test coincide con la enumeración exacta sobre
los $2^n$ perfiles.

Una decisión resultó crítica. Con la actualización **secuencial** (condicionar solo
en el último test) la separación es esencialmente nula: ambos greedys rinden igual.
La ventaja solo aparece con la actualización de **toda la historia**, porque es ahí
donde el conteo habilita las deducciones cruzadas que el binario no puede hacer.
Esto valida empíricamente la insistencia de Francisco en condicionar el posterior
en el historial completo, no en la última consulta. Todo lo que sigue usa el régimen
full-history para ambos competidores.

## Dónde aparece la separación

### Prevalencia: se enciende en $\rho \approx 0.15$ y crece

Barrido de prevalencia con $N=12$, $B=3$, $G=4$ (prior heterogéneo centrado en
$\rho$):

| $\rho$ | separación | aug gana |
|---|---|---|
| 0.05 | −0.54% | 62% |
| 0.10 | −0.48% | 50% |
| 0.15 | +2.15% | 100% |
| 0.20 | +1.71% | 62% |
| 0.25 | +2.57% | 100% |
| 0.30 | +1.80% | 75% |
| 0.35 | +7.43% | 100% |
| 0.40 | +7.41% | 100% |

A prevalencia baja casi todos los pools salen conteo-ceros, el conteo no añade nada
sobre el binario y la separación es nula (incluso levemente conteo-cero por ruido). A
partir de $\rho \approx 0.15$ la separación se vuelve conteo-no-cero y consistente (el
aumentado gana en el 100% de las instancias) y crece con la prevalencia. Esto
confirma el borde inferior del régimen que Francisco señala; el borde superior, donde
la utilidad colapsa por falta de limpios, queda por encima de 0.40 y no se barrió.

### Tamaño de pool: un salto en $G > 3$

Barrido de $G$ con $\rho=0.25$, $N=12$, $B=3$, 24 instancias por punto (error
estándar entre paréntesis):

| $G$ | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| separación | 0.55% | 1.31% | 4.26% | 3.22% | 3.98% | 3.66% |
| (SEM) | (0.27) | (0.85) | (0.93) | (1.04) | (1.12) | (1.09) |

Hay un cambio de régimen nítido entre $G=3$ y $G=4$: la separación es chica con pools
de hasta tres ($\le 1.3\%$) y salta a alrededor de $4\%$ con cuatro o más, donde se
estabiliza. La distancia entre $G=2$ y $G=4$ es de más de tres errores estándar. Esto
confirma la hipótesis central de Francisco de que la ventaja del conteo necesita
$G>3$. El mecanismo es directo: con pools de hasta tres, el conteo es casi siempre
0 o 1 y coincide con el binario; con cuatro o más aparecen conteos intermedios
(2, 3, …) que el binario no distingue de "conteo-no-cero", y ahí el conteo paga.

## Una salvedad honesta sobre usar greedy para medir la separación

La separación limpia es entre óptimos, y para los óptimos vale siempre
$U^D_A \ge U^D$. Para el greedy no: el greedy aumentado **no** domina al greedy
clásico instancia por instancia. En la validación, el clásico superó al aumentado en
algunos casos (peor violación 0.18 de utilidad). La razón es que ambos son
heurísticas miopes, y más información no garantiza una mejor decisión cuando la regla
no la usa de forma óptima. La falla de dominancia se concentra a prevalencia baja,
donde no hay nada que deducir; en el régimen donde la separación es clara
($\rho \ge 0.15$) el aumentado gana en casi el 100% de las instancias. Esto conecta
con el rompecabezas de por qué el greedy le ganó al aprendizaje por refuerzo, y es un
recordatorio de que el barómetro honesto compara óptimos donde se puede y trata al
greedy como una cota inferior ruidosa de la separación verdadera.

## Lo que queda

El experimento corre con greedy a $N=12$, dentro del alcance del posterior por conteo
exacto. Las dos extensiones naturales son escalar a $N=30$–$50$ con el muestreo de
Gibbs full-history (que captura las mismas deducciones a costo acotado) para
confirmar que el salto en $G>3$ y el encendido en prevalencia intermedia persisten, y
medir la separación entre óptimos en una rejilla pequeña de su régimen para anclar el
proxy de greedy contra la señal limpia. Ambas materializan el plan de Francisco:
empezar por lo empírico, verificar la corrección, y solo entonces escalar.
