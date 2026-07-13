# El modelo realista de pruebas: ideas para la §3

Notas de exploración para la rama del modelo realista (la §3 del notebook
`19_avances_post_sesion.ipynb`). Hoy el modelo idealiza el conteo como exacto; una
prueba real —qPCR, biomarcadores— lo entrega con ruido. La pregunta es si la
separación de la §1 (dinámico aumentado vs óptimo estático) sobrevive a ese
ruido, y cuánto aguanta.

## El giro: el esquema casi no usa el conteo

La estrategia dinámica de la §1 nunca lee el conteo exacto. El binary search
solo necesita un bit por prueba: si el bloque está saturado (todos infectados)
o no. En cada corte pregunta "¿esta mitad está toda infectada?" y con eso decide
hacia dónde seguir; el conteo exacto lo descarta.

Eso invierte la preocupación. La pregunta no es si la separación sobrevive al
ruido del conteo, sino cuánto ruido aguanta un solo bit —saturado sí o no—
antes de caerse. Y esa discriminación es fácil: distinguir conteo = tamaño de
conteo = tamaño − 1 es un gap de una persona. La separación debería ser
robusta, y cuantificar cuánto aguanta es en sí el resultado publicable: se pasa
de "ojalá sobreviva" a un umbral de ruido con teorema.

## Cuatro modelos de ruido, de más limpio a más realista

**A. Bit-flip sobre saturado/no saturado.** El modelo mínimo: cada consulta de
saturación se equivoca con probabilidad ε. La búsqueda acierta si sus log₂G
pasos aciertan, así que la utilidad dinámica se vuelve
u·(1−(1−q)^kG)·(1−ε)^(log₂G). Comparada con B·u·q da un umbral en ε: para ε
pequeño el castigo es ≈ 1 − ε·log₂G, de modo que grupos no gigantes lo toleran.
Es la extensión analítica directa de la §1 y da un teorema-juguete inmediato.

**B. Conteo continuo con umbral (el que huele a qPCR).** Se observa el conteo
con ruido gaussiano de desviación σ y se decide "saturado" por umbral. La
discriminación relevante es siempre G contra G−1 (¿hay una persona sana
diluyendo?), gap = 1, así que el error por paso es ≈ Φ(−1/(2σ)), independiente
del tamaño del bloque. Eso da un umbral en σ limpio: la separación aguanta
mientras σ sea chico frente a una persona. Es el puente natural a la §1 y el más
fácil de graficar: una curva de ventaja contra σ con el umbral marcado.

**C. El test devuelve una distribución sobre conteos.** En vez de un número,
evidencia suave: una verosimilitud P(observado | r). El marco DAPTS se
generaliza: la fibra dura (perfiles consistentes con el conteo exacto) se vuelve
una fibra suave donde cada perfil pesa según la verosimilitud. La inferencia
existente se extiende casi sola —en vez de una restricción dura por prueba, un
factor blando por prueba— y el binary search se convierte en una prueba de
hipótesis secuencial (tipo SPRT): si no hay certeza, se vuelve a medir.

**D. El mecanicista de verdad (para calibrar con el paper de Francisco).** Cada
infectado aporta carga viral Lᵢ ~ log-normal, con varianza enorme en la
realidad; la señal del pool es Σ Lᵢ y el Ct es una función de esa suma más ruido
de medición. Aquí el conteo ni siquiera es identificable: dos infectados de
carga baja se ven como uno de carga alta. Es el modelo honesto y el más difícil;
sirve para calibrar σ en B y C, no para el primer teorema.

## Tres conexiones con lo ya construido

Con la curva de resolución (D2): un conteo ruidoso es un canal estocástico entre
el binario y el conteo exacto. El resultado del 84.5% se generaliza a cuánto del
valor sobrevive a un canal de fidelidad dada; la §3 deja de ser una rama suelta
y se vuelve el eje de resolución con ruido, la cuarta perilla del mapa.

Con la inferencia (Gibbs corregido): la fibra suave es un cambio de una línea
conceptual, constraint dura a factor de verosimilitud. La descomposición por
componentes y el sampler con la corrección de Hastings extienden directo. El
ejemplo laminar y de cadena de la §2 también: los conteos por capa dejan de
restarse exacto y se propagan como mensajes suaves, sigue siendo
forward-backward pero con ruido.

Con el riesgo: el ruido introduce el falso-limpio —certificar sano a alguien
infectado—, el error peligroso para tamizaje. Eso reintroduce el objetivo de
riesgo/CVaR ya explorado (P(bienestar = 0)); la separación deja de ser solo
cuánta utilidad y se vuelve utilidad segura, que es lo que le importa a un
ministerio de salud.

## La perilla nueva: re-medir

Con ruido se puede volver a probar el mismo pool para reducir incertidumbre. Es
un trade-off de presupuesto que antes no existía: gastar pruebas en cubrir más
gente contra gastarlas en medir con más precisión a la misma. Es una pregunta
genuinamente nueva y del tipo budget-allocation con dos ejes. El SPRT del modelo
C es exactamente la política óptima de cuántas veces re-medir antes de decidir.

## Orden recomendado

Empezar por A y B: son la §1 con un parámetro de ruido encima, dan umbrales
cerrados (ε\* y σ\*) y una figura de ventaja contra ruido que se lee sola. Es el
resultado corto y publicable, y encaja con lo que pidió Francisco: no mover más
variables, agregar una (el ruido) y mostrar el umbral. C queda como la
generalización del marco (la fibra suave, que reconecta con la inferencia) y D
para cuando llegue el artículo de biomarcadores.

Primer entregable concreto: el modelo B como celda ejecutable de la §3 —la curva
de ventaja contra σ con el umbral marcado, reusando `util_estatico` y
`util_dinamico` de la §1 más un factor de error de saturación—. Es el gemelo con
ruido de la figura que ya existe, y probablemente el gancho más fuerte para la
próxima sesión.
