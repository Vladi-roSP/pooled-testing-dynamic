# Estado del proyecto

El esquema augmented pasó de tener números atacables a tener inferencia y
evaluación correctas y probadas, con el cómputo pesado listo para el clúster y un
borrador de paper.

## Lo que se corrigió

Nueve arreglos, cada uno con su consulta. Los cuatro de fondo son los pesos de rama
([[Pesos de rama y el sesgo del 17 por ciento]]), la cosecha de limpios
([[Cosecha de limpios deducidos]]), la ergodicidad de Gibbs
([[Gibbs no era ergódico]]) y el fallback silencioso de la inferencia, que ahora
lanza error. Los demás fueron imports de solver mal colocados, un crash por
empates, un test que no corría como script, un docstring falso, dos bugs del
entorno de RL, y la regresión de rendimiento del exacto a escala (resuelta con el
dispatch por tamaño).

## Lo que se mejoró

El generator de Gibbs se reescribió para ser ergódico
([[Gibbs — muestreo ergódico]]). Se dejaron listos los scripts de cómputo pesado en
`augmented/compute_center/`. Se escribió el borrador del paper, su guía en español,
y `hierarchy_experiment.py` con el código corregido.

## Los números

La cadena de [[La jerarquía de óptimos]] se cumple en cientos de instancias, y la
ventaja del conteo sobre el binario crece con la escala: +0.63% en N = 3, +3.97% en
N = 5, +5.07% en N = 7. Es la curva central del paper y respalda el mecanismo de
[[Por qué dinámico y por qué conteo]]: el conteo paga vía mejores posteriores sobre
un horizonte más largo. [[El descubrimiento del horizonte]] reencuadró después esa
curva: la perilla que gobierna el beneficio es el horizonte B, no la escala.

## Lo nuevo (julio 2026)

La reunión con Francisco del 2 de julio reorganizó el paper alrededor de tres
direcciones, con la curva de resolución como núcleo, ya implementado y medido:
distinguir 0 / 1 / ≥2 captura entre el 85% y el 100% del beneficio del conteo en
el régimen exacto. El mapa completo y los resultados están en
[[Las tres direcciones]] y las notas de la carpeta Resultados.

## La línea de certificados (6–7 de julio)

Dos días que cambiaron el frente. Primero, la auditoría encontró y corrigió un
segundo defecto del Gibbs — irreducible pero con estacionaria sesgada por falta
del factor de Hastings, probado con matriz de transición exacta
([[Gibbs y el equilibrio detallado]]). Después se abrió la línea D3 en código:
U_PI y la primera cota penalizada del problema (`certificates.py`, validada
instancia por instancia), con tres hallazgos — el greedy es mucho mejor de lo
demostrable, el apriete de la penalización es un fenómeno de horizonte, y la V̂
simple vence a la sofisticada porque el adversario explota su error de
independencia ([[La primera cota penalizada]]). La figura que une las tres
perillas con su capa certificada está en [[El mapa con garantías]], la dirección
propia que todo esto sostiene en [[El cuarto eje — el certificado computable]],
y su traducción a producto en [[La demo de flota]] (78% certificado contra 46%
del muestreo aleatorio en n=50).

## El estado de las consultas

Todo verde, verificado en aislamiento: 21 de solvers, 79 del principal, 12 de
correctitud, 2 de RL, más los de otros módulos y los 25 escenarios de Gibbs.

## Lo que sigue

Compartir el draft con Francisco, decidir el merge de la rama, correr los scripts
del clúster para los números a gran escala, y el ángulo de datos reales de counting.
Ver [[Para Francisco]] y [[Preguntas abiertas]].
