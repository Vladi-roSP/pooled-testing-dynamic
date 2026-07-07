# Plan para la sesión del jueves 9 de julio con Francisco

7 de julio de 2026. Este documento fija dos cosas: la dirección de investigación propia que se va a proponer, y el trabajo concreto de martes y miércoles para llegar a la sesión con resultados fuertes.

Insumos: las tres direcciones de Francisco del 2 de julio (`Vault-GroupCounting/Resultados/Las tres direcciones.md`), el memo del norte (`norte-5-direcciones.md`), el análisis de capacidad del 6 de julio, y la corrección del sampler de Gibbs completada el 6 de julio.

## La dirección elegida, explicada de una vez

Francisco resumió el paper como un mapa con tres perillas: el presupuesto de tests B, el traslape de los grupos K, y qué tan detallada es la respuesta de cada test (la resolución). El mapa dice en qué combinaciones de perillas la información extra del conteo vale algo y en cuáles no.

La dirección propia agrega el eje que ese mapa no nombra: el cómputo. Saber que la información vale no sirve si nadie puede calcular la estrategia que la aprovecha. La pregunta nueva es: de todo el valor que el mapa promete, ¿cuánto puede reclamar de verdad un algoritmo que corre en tiempo razonable? ¿Y cómo se demuestra, sin conocer el óptimo, que lo reclamado está cerca de lo posible?

La herramienta para responder es el certificado. La idea, en llano: como el óptimo verdadero es incalculable a escala, se le acota por arriba con un adversario imaginario que conoce el futuro (la cota "hindsight" o de información perfecta, que llamamos U_PI). Nuestra estrategia real da la cota por abajo. El cociente entre ambas es una garantía: "esta estrategia logra al menos X% de lo mejor posible". Hoy esa garantía es floja porque el adversario que ve el futuro es demasiado fuerte: certifica solo 58% en instancias de 50 personas. La técnica estándar para apretarla —dualidad con penalización, Brown–Smith–Sun— le cobra al adversario una multa por usar información que no debería tener. Nadie ha aplicado esa técnica a este tipo de problema. Ese es el hueco que se reclama como propio.

Con este lente, todo lo ya hecho se acomoda: D3 es el certificado mismo; D1 dice cuándo la inferencia que el certificado necesita es computable (la dureza #P, el mixing del Gibbs); D2 es el certificado aplicado a la resolución (el test barato de tres niveles captura 84.5% del valor — y ahora se puede certificar); el hallazgo del horizonte es el eje B del mismo objeto; y el arreglo del Gibbs es lo que vuelve confiable cualquier certificado a n=50. La startup vende exactamente este objeto: el memo del norte lo llama "motor con garantía" y el borrador de YC ya declara la dualidad penalizada como el trabajo activo.

La frase para la sesión: "Tu mapa dice cuándo contar vale; yo quiero caracterizar cuánto de ese valor se puede reclamar y certificar con cómputo finito, como función de tus tres perillas."

Dos decisiones de alcance. La aplicación a flotas de agentes de IA (los evals por lotes son pools; los graders son canales con error) se usa como demostración de diez minutos si la conversación va hacia la industria, no como la dirección central; su semilla teórica —conteos con ruido, una cuarta perilla que nadie del grupo ha tocado— se menciona como pregunta abierta. Y el encuadre de teoría de la información (tasa–distorsión) se reduce a un párrafo del paper.

## Cómo se presenta el jueves (el arco)

1. Abrir con rigor: el bug de equilibrio detallado en el Gibbs — encontrado, probado con matriz de transición exacta, corregido, con toda la suite en verde. Establece nivel y de paso responde las dos indicaciones técnicas que él dejó (medir en TV, estudiar el mixing).
2. Seguir con su número favorito mejorado: la cota de D3 apretada por penalización.
3. Mostrar el mapa con certificados: una figura donde sus tres direcciones aparecen como rebanadas de un solo objeto.
4. Solo entonces, decir la frase de la dirección. A esa altura no es una propuesta: es la conclusión de lo que acaba de ver.
5. Cerrar con tres preguntas abiertas para trabajar juntos. Eso lo convierte en coautor de la dirección, no en su evaluador.

## Martes — Workstream 1: apretar la cota (el resultado estrella)

Por qué: es el único resultado nuevo que mueve la aguja el jueves, y es exactamente el músculo que el análisis de capacidad pide construir.

Qué se hace, en orden:

1. U_PI limpio. Hoy la cota hindsight existe solo como prototipo en notas (tabla en `paper/lineas_research_francisco.md`). Se implementa como módulo con tests y CSV, y se valida contra el óptimo exacto en n∈{4..7}: por construcción debe cumplirse U_PI ≥ óptimo en toda instancia.
2. U_pen, la versión penalizada. La multa de cada paso es la diferencia entre lo que el adversario aprende viendo el futuro y lo que habría aprendido sin verlo, medida con una función de valor aproximada V̂ (el candidato natural es el valor miope que el greedy ya computa). La teoría garantiza que cualquier V̂ da una cota válida; una V̂ buena la da apretada. En instancias chicas (n≤6, B≤3) el problema interno se resuelve por fuerza bruta, así que el apriete se verifica contra el óptimo exacto.
3. El entregable: una tabla con dos columnas — fracción certificada con hindsight vs con penalización — para n∈{4..7}. La meta es subir de ~0.84 (n=7) hacia ~0.95.

Control de calidad automático: si en alguna instancia exacta sale U_pen < óptimo, la penalización está mal implementada. El experimento se audita solo.

Stretch (si el martes rinde): una penalización descomponible que funcione a n=50 y dé el primer número mayor al 58% actual. Si no sale, no pasa nada: el apriete demostrado en pequeño más el camino a escala es suficiente para el jueves.

## Miércoles — Workstream 2: el mapa con garantías

Una sola figura. Ejes: presupuesto B contra resolución (cap); si el tiempo alcanza, un panel extra con K=1 contra pools libres. En cada punto, dos números: la fracción real que logra el greedy (contra el óptimo exacto, computable en estas escalas) y la fracción que el certificado garantiza. Todo reúsa código existente: `solve_optimal_dapts(cap)`, las instancias de la curva de resolución y las del experimento de horizonte. Medio día.

Es la imagen que une todo sin argumentos: sus tres direcciones, un solo objeto.

## Miércoles tarde — Workstream 3: demo y rigor

Dos piezas cortas:

1. La demo de flota (2–3 horas): una corrida n=50 presentada en lenguaje de agentes: "50 componentes, 12 evals por lotes; estos 31 componentes quedan certificados limpios; la asignación del presupuesto fue al menos X% de la óptima". Un printout de terminal y una lámina. Se enseña solo si él lleva la conversación a la industria.
2. Rigor (30 min): commit y push del fix de Gibbs con su test de regresión, y añadir la "corrección 3" a `paper/correcciones_gibbs.md` junto a las dos de junio.

## Jueves en la mañana — los activos (2 horas)

Una página o mini-deck con cinco piezas: el fix de Gibbs y su prueba; la tabla del apriete; el mapa con certificados; la frase de la dirección; y las tres preguntas abiertas — qué V̂ es la penalización correcta, conteos con ruido como cuarta perilla, y el mixing time del Gibbs como función de K.

## Si algo falla

Si el miércoles en la noche la penalización no aprieta, la sesión sigue de pie: el fix de Gibbs probado, U_PI scriptado y validado, el mapa con la cota hindsight, y la dirección. El plan no depende de que el resultado estrella salga.

## Estado de ejecución

- [x] WS1.1 — U_PI como módulo + validación (certificates.py, 9/9 tests)
- [x] WS1.2 — U_pen penalizada + verificación exacta en n≤6
- [x] WS1.3 — tabla hindsight vs penalizada (data/certificates_small_n.csv, 106 instancias)
- [ ] WS1.stretch — U_pen descomponible en n=50 (opcional; el camino queda demostrado en pequeño)
- [x] WS2 — figura del mapa con certificados (figures/certified_map.png; en B=3 el canal de 3 niveles certifica igual que el conteo exacto)
- [x] WS3.1 — demo de flota n=50 (demo_fleet_certification.py: 78% certificado vs 46% aleatorio)
- [x] WS3.2 — commit + push (308e7ff) y corrección 3 en paper/correcciones_gibbs.md
- [x] Una página de la sesión (Desktop/ASE/sesion-jueves-una-pagina.md)
- [ ] Jueves — ensayo del arco y demo en vivo
