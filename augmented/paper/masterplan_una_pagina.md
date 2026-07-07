# Sesión del 9 de julio — una página

## 1. Rigor: el Gibbs necesitaba una segunda corrección, y está hecha

La reescritura de junio hizo la cadena irreducible pero dejó un defecto de equilibrio detallado: la propuesta por caminos alternantes es asimétrica y la aceptación era Metropolis puro. La cadena convergía, estable en todas las semillas, a una posterior equivocada (TV 0.067 en el contraejemplo mínimo; 6.7 puntos de error en una marginal). Se probó con la matriz de transición exacta de la cadena —enumerando todas las ramas del generador—, se corrigió con el factor de Hastings por camino espejo, y la matriz corregida da TV 0.000000 en las cinco topologías auditadas. Suite completa: 79/79. De paso quedan respondidas las dos indicaciones técnicas de la reunión pasada: las distancias se midieron en TV, y ahora hay una cadena correcta sobre la cual estudiar mixing.

## 2. El certificado se puede apretar: primera cota penalizada

La cota hindsight U_PI certifica poco porque el adversario que ve el futuro es demasiado fuerte. Se implementó la primera cota penalizada (Brown–Smith–Sun) para este problema, con el problema interno exacto y validada instancia por instancia contra el óptimo (106 instancias, cero violaciones):

| config | greedy/OPT (real) | certificado U_PI | certificado U_pen |
|---|---|---|---|
| n=4 B=2 G=2 | 0.984 | 0.794 | 0.843 |
| n=5 B=2 G=3 | 0.986 | 0.702 | 0.731 |
| n=6 B=2 G=3 | 0.981 | 0.631 | 0.683 |
| n=5,6 B=3 | 0.93 | 0.76–0.82 | sin cambio |

Dos lecturas. El greedy es mucho mejor de lo que se puede demostrar (real ~0.98, certificado ~0.7): el cuello de botella es la demostración, no el algoritmo. Y el apriete es un fenómeno de horizonte: la penalización con V̂ miope funciona en B=2 y se apaga en B=3 — el mismo patrón que la ley del lookahead (99% → 40% → 16%). La V̂ correcta debe mirar tan lejos como el horizonte.

## 3. El mapa con garantías

Figura `figures/certified_map.png` (n=5, G=3, prevalencia 0.25–0.65, 12 instancias): tus tres perillas en un solo objeto. Por cada punto (B, cap), la fracción del valor que es real (curva de resolución, solo computable en n chico) y la que es certificable a cualquier escala. La banda entre ambas es el programa de investigación. Dos datos que la figura deja ver: la fracción certificada crece con el horizonte (0.58 en B=1 → 0.85 en B=3), y en B=3 el canal de tres niveles certifica exactamente lo mismo que el conteo completo (0.85 en cap=2 y cap=3, contra 0.79 del binario) — la versión certificada del resultado del 84.5%.

## 4. La dirección

Tu mapa dice cuándo la información vale. Yo quiero caracterizar cuánto de ese valor se puede reclamar y certificar con cómputo finito — el gap entre valor de información y valor computable, como función de tus tres perillas. D3 es el certificado; D1 dice cuándo su inferencia es computable; D2 es el certificado aplicado al canal.

(Si la conversación va a la industria: demo en vivo, 30 segundos. Una flota de 50 componentes, 10 corridas por lotes: el motor certifica 31 limpios sin un solo falso limpio y garantiza ≥ 78% del óptimo incalculable; el muestreo aleatorio con el mismo presupuesto logra 46%. `demo_fleet_certification.py`.)

## 5. Tres preguntas para trabajar juntos

1. ¿Cuál es la V̂ correcta para la penalización? El hallazgo empírico: el potencial simple (lineal en marginales exactas) certifica mejor que el greedy-a-futuro, porque el adversario interno explota el error de independencia de la V̂ sofisticada — el independence gap atacando al certificado. ¿Existe una V̂ con profundidad d(B)?
2. Conteos con ruido: si el test reporta r con error (el grader se equivoca), ¿cómo se degrada la escalera de resolución? Cuarta perilla del mapa; nadie la ha tocado.
3. Mixing time del Gibbs corregido como función del grado K del hipergrafo — el puente entre D1 y la capa de inferencia del certificado.
