# Sesión con Francisco — Harvard Station (2026-08-11)

**Fuente:** `TRANSCRIPCION_HARVARD_STATION_ACTIVE.md` (diarización con Hablantes A/B/C **no confirmados**; la etiqueta A mezcla voces del equipo y de Francisco por ruido de estación). Disciplina de atribución: se cita como "la conversación plantea…"; a Francisco solo se le atribuye lo inequívoco por contexto de rol. Nota de contexto (persona A): la reunión acordada para el día siguiente [35:47–36:34] resultó de carácter social, sin contenido de trabajo; el despacho de esta acta se realizó el 2026-08-18.

## Resumen ejecutivo

El equipo entrega el encargo de la sesión anterior (2026-08-02): el colapso de V(T)=E_R[B(R)] con cómputo exacto. La conversación valida la obstrucción ("el score en esencia también es miope"), la refina — el score colapsado **ni siquiera es submodular: es exactamente aditivo**, V(S)=Σuᵢqᵢ — y añade suboptimalidad en B=1 con q<0.5. Se reconoce la mitad buena (impulsa el primer movimiento del greedy). Sobre esa base la conversación desarrolla **la directriz nueva: un score de dos componentes** — el valor V(T) de siempre más un **costo C(T) = número esperado de pruebas que greedy necesita para extraer la utilidad viva** — estimable por Monte Carlo (muestrear el posterior, correr greedy muchas veces), con regla de poda por presupuesto restante. Simplificación de regímenes encargada: primero u≡1; después q homogénea con u heterogénea. Framing pedagógico: el score ideal es la Q-table de Bellman (Q*(s,a), "difícil sino imposible"); el costo es el proxy practicable.

## D1 — Entrega del encargo y validación de la obstrucción [00:00–06:08]

El equipo presenta la implementación (ejemplo q=0.15, u=1, n=4) y el hallazgo: "la puntuación que te da el score siempre va a ser el mismo que las probabilidades previas… lo que una rama te sube, otra rama te lo va bajando… **este score en esencia también es miope**" [01:02–01:55]. La conversación lo eleva: "tú tienes una set function… el valor del conjunto S es igual a la suma de uᵢqᵢ… **ni siquiera submodular, es exactamente aditivo**" [03:40–04:56], y añade: el primer paso "siempre va a maximizar el tamaño del grupo… y es subóptimo en el caso B=1… si q<0.5 este algoritmo va a poner un montón de personas" [04:56–06:08]. Mitad buena reconocida: "sí puede ayudar a que el greedy haga ese primer movimiento" [01:02–01:55, 03:40–04:56]. Diagnóstico compartido: "algo que está faltando aquí es **incorporar el presupuesto restante**" [06:08–06:49].

## D2 — Directriz nueva: score de dos componentes, con greedy como oráculo de costo [07:03–09:03, 27:39–34:48]

"Cuando tú haces una prueba tiene dos puntajes: el **valor** de la prueba… y una medida del **costo en pruebas** para poder hacer charge de su utilidad" [07:03–08:07]. Definición del costo: "si de ahora en adelante hago pruebas óptimas para este subconjunto, en promedio cuántas necesito para llegar a una hoja terminal… las ramas del árbol tienen valor y tienen largo también" [08:07–09:03]; operativamente: "**el costo de T es cuántas pruebas usa greedy hasta finalizar**… haces un muestreo del posterior y corres greedy mil veces y ves en promedio cuánto cuesta" [30:27–32:40]. Regla de uso: poda por factibilidad — "si requiero 10 pruebas y mi presupuesto es 5, no puedo hacer eso" [07:03–08:07]; "si el presupuesto es demasiado chiquito lo vamos quitando y no vamos a tomar esa opción" [33:50–34:36]. El gran pero, dejado explícito: "**¿cómo hacemos el cómputo de esto?** Quizá en el caso de u=1 es más fácil… el tamaño de grupo greedy es una expresión cerrada, función cóncava… quizá exista una expresión cerrada" [34:36–35:45].

## D3 — Simplificación por regímenes [20:57–21:42, 34:53–35:45]

"Un paso intermedio sería enfocarse en el problema sin utilidades heterogéneas… una limitación inicial de que u=1 para todo mundo, podemos empezar por esa respuesta" [20:57–21:42]. Segundo régimen: "q igual a algo y las utilidades diferentes — mantener heterogeneidad sin llegar al caso muy complejo" [34:53–35:45].

## D4 — Framing RL/Bellman (pedagógico, no headline) [24:06–27:39]

La conversación enmarca el score ideal como Q-table: "te dice, estando en este estado y tomando esta acción, el valor óptimo restante… la ecuación de Bellman" [25:23–26:33]; "esto no suele ser difícil sino imposible de obtener. **Todo vuelve a que necesitamos alguna medida del costo**" [26:33–27:39]. RL permanece "no como headline" (§34); esta mención no habilita línea RL (§31).

## Aportes del equipo reconocidos en la mesa

- El caso n=5, q=0.3, B=3, G=2 con el árbol del óptimo, verificado "en todo el rango de probabilidades desde 0.6 hasta 0.9" [17:21–18:58] (cifra verificada del repo: coincidencia rollout-óptimo desde p ≥ 2−√2 ≈ 0.586 en la rejilla p∈{0.50,…,0.90}).
- La propuesta de normalizar por pruebas restantes — la "descontada" — salió del equipo [06:49–07:03, 23:05–24:06] y la conversación la formalizó como el par valor/costo.
- El polinomio del árbol por expansión binomial [21:42–22:56] (verificado: 1.011 = q(3q²−3q+4) en q=0.3).
- El vocabulario "utilidad viva" quedó adoptado en la mesa [09:05–09:16].

## Preguntas de §34 respondidas en esta sesión

- **(2)** ¿sanos identificables o utilidad acreditable con presupuesto residual? → **Respondida:** la dimensión del presupuesto es imprescindible; forma acordada: valor + costo-en-pruebas.
- **(7)** ¿descuento fijado por log G o calibrado? → **Respondida en forma nueva:** el costo se **mide** (rollout de greedy, Monte Carlo sobre el posterior); esperanza de forma cerrada en homogéneo.
- **(6)** parcialmente: la regla propuesta es greedy sobre las opciones factibles (costo ≤ presupuesto) con dos puntajes.

## Compromisos y pendientes

- Pendiente del equipo declarado en sesión: "faltará más la escritura de por qué el colapso" [26:33–27:39] — **cumplido post-sesión** (Hechos 1 y 2 redactados, dos versiones etiquetadas cada uno).
- Compromiso fechado del 2026-08-02 (evaluar V(T) en el ejemplo con cómputo exacto): **CUMPLIDO en esta sesión.**
- Siguen vigentes: reformulación de la Conjetura C (P21-A8); cotejo de Francisco del teorema estático p>1/2 en el paper.
