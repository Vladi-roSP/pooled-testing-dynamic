# Preguntas abiertas

Las líneas que quedan vivas, ordenadas de la más teórica a la más aplicada. Llevar
una de estas, bien planteada, a la conversación con Francisco vale más que repetir
lo ya hecho; ver [[Para Francisco]].

## La penalización correcta (la V̂ con profundidad d(B))

La nueva pregunta central de la línea de certificados. Cualquier V̂ da una cota
válida ([[Cotas por relajación de información]]), pero la evidencia de
[[La primera cota penalizada]] impone dos condiciones a la V̂ buena: ser
insesgada donde el adversario mira (la V̂ greedy-a-futuro pierde contra el
potencial simple porque el adversario explota su error de independencia, ver
[[El independence gap]]) y mirar tan lejos como el horizonte (el apriete se
apaga en B=3 con V̂ miope, calcando la ley del lookahead de
[[El hueco del greedy — miopía y lookahead]]). ¿Existe una familia de V̂ con
profundidad d(B) cuyo problema interno se descomponga, para dar el primer
certificado apretado en n=50? Es la pregunta con más palanca simultánea sobre
el paper y sobre la empresa.

## Submodularidad adaptativa del greedy

La pregunta teórica central. Si el valor por paso bajo conteo fuera submodular
adaptativo (en el sentido de Golovin–Krause), el greedy heredaría una garantía
del tipo 1 − 1/e respecto al óptimo. Es el terreno de Francisco y conecta con su
formulación de super-nodos y con [[El independence gap]]: el obstáculo es que el
conteo obliga a mirar la distribución completa, no un resumen escalar.

## Inferencia aproximada con garantías

[[Dureza P de la inferencia]] cierra la puerta al cálculo exacto general, pero deja
abiertas preguntas finas: ¿hay esquemas de aproximación con garantías para clases
naturales de diseños de test? ¿La propagación de creencias es asintóticamente
correcta en diseños aleatorios dispersos? ¿Se pueden diseñar las políticas de test
junto con la inferencia para mantenerla tratable?

## El conteo con ruido (datos reales de counting)

La pregunta aplicada y la más relevante para el track de impacto social. El modelo
idealiza el conteo como exacto, pero la counting lo entrega a través de un cycle
threshold ruidoso. ¿Cómo se comporta la ventaja del conteo cuando el conteo se
observa con error? Calibrar el modelo contra datos reales de prevalencia o de
carga signal es lo que convertiría el paper en un trabajo de despliegue, no solo de
algoritmos.

## Los números a gran escala

Lo menos abierto y lo más inmediato: correr los scripts de
`augmented/compute_center/` para tener la ventaja del conteo a N grande y el RL
entrenado de verdad. No es investigación nueva, es ejecución pendiente; ver
[[Estado del proyecto]].
