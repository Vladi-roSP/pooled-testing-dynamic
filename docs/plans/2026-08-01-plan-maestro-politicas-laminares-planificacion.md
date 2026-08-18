# PLAN MAESTRO — Políticas laminares con planificación para Dynamic Augmented Group Counting

**Estado:** plan maestro vigente, aprobado por A y B el 3 de agosto de 2026 (norte científico y operativo del proyecto). Resultado de revisión adversarial cruzada.
**Fecha de corte del repositorio:** 1 de agosto de 2026.
**Documentos que retira** (nota de supersede al inicio de cada uno; se conservan como historial y trazabilidad — ninguna tarea incompleta desaparece: cada una queda mapeada a un milestone con gate y responsable):
`docs/plans/2026-07-21-plan-semanal-laminar-milp.md` · `docs/plans/2026-07-27-tensor-greedy-laminar.md` · `augmented/paper/masterplan.md` · `augmented/paper/masterplan_una_pagina.md`
**Personas:** A = Vladimir (teoría propedéutica + diseño del objetivo) · B = Héctor (implementación + infraestructura) · **Soporte de investigación asistida por IA** (búsqueda inicial, candidatos, crítica adversarial, harness, organización bibliográfica; **A valida fuentes primarias e hipótesis y B valida todo código antes de incorporar cualquier resultado — la responsabilidad epistemológica es siempre de A y B**).
**Atribución y procedencia:** la dirección central del proyecto se atribuye a Francisco a nivel global (testimonio directo de A), no punto por punto. La conversación de Boston College recapitula y valida esa dirección ante una interlocutora externa y aporta como entrada distintiva principal la vía submodular con sus relajaciones convexas; sus citas se usan como "la conversación plantea…", sin atribución línea a línea porque las voces no están confirmadas. Las formalizaciones y extensiones posteriores son del equipo. **La inclusión de una línea en este plan no implica que haya sido solicitada expresamente por Francisco**; la procedencia de cada empuje mayor está en §0-bis.

---

## 0-bis. Procedencia de la dirección

**(a) Espina del programa** — dirección de proyecto atribuida a Francisco a nivel global, no punto por punto; recapitulada y validada en la conversación de Boston College: estudiar el valor del setting dinámico aumentado (extensión del paper EC: +dinámico, +aumentado) · la separación frente a estático/binario como punto de partida y prueba mínima, con el mecanismo de alta actividad ($R{=}G{-}1$ ⟹ búsqueda binaria) · comparaciones anidadas entre clases de políticas (origen de §7–§8) · restricción laminar y conteo del complemento gratis · objetivo greedy por iteración con planificación incorporada · criterio de validación: recuperar el ejemplo motivador · contexto del paper hermano ($q<1/2$ ⟹ el pooling binario no paga en la celda estática).

**(b) Entrada distintiva principal de la conversación de Boston College:** ¿admite el objetivo estructura submodular? — incluidas las relajaciones convexas asociadas (→ §20, §21, §34). **Interés prioritario de A.**

**(c) Formalización instrumental del equipo** (necesaria para ejecutar (a)): modelo normativo con acreditación parcial · tensor, átomos y Lema A · scorers $S_0$–$S_3$ con su linaje (§14) · rollout como oráculo · acid test como gate · falsificador · descomposición de pérdidas · disciplina de claims y reproducibilidad · ex ante/ex post · celda dinámico-binaria — diseño del equipo que formaliza la comparación dinámica/conteo planteada en la conversación (responde a C4).

**(d) Extensiones propuestas por el equipo** (agenda propia, subordinada): frontera $B{=}2/B{=}3$ · atlas extendido · certificados · escala · líneas de §30.

La itemización traza al programa y a la conversación, no a peticiones individuales de una persona. Las extensiones (d) son agenda válida del equipo, pero no constituyen evidencia de cumplimiento de (a); el criterio de poda operativo está en §1.

---

## 0. Resumen ejecutivo

**Dirección central:** diseñar una política greedy laminar que incorpore valor futuro, verificar que recupera el ejemplo de separación, compararla contra rollout y determinar qué garantía — submodular u otra — admite.

> **Decisión rectora:** la separación demuestra que los conteos pueden crear valor; la laminaridad hace computable la información que lo produce; la planificación determina si una política logra capturarlo.

El problema científico no es otra heurística ni más experimentos: es aproximar el mejor árbol dinámico aumentado mediante políticas que (1) exploten conteos, (2) condicionen en el historial completo, (3) preserven inferencia exacta, (4) incorporen valor futuro, (5) sean computables, (6) admitan garantía o caracterización honesta.

Cuello de botella actual: **diseñar y falsificar una función local que aproxime el valor de rollout a menor costo.**

$$\boxed{\text{modelo} \to \text{tensor} \to \text{Lema A} \to \text{objetivo} \to \text{acid test} \to \text{rollout} \to \text{falsificación} \to \text{garantía} \to \text{atlas} \to \text{escala}}$$

Las etapas cierran por gate, no por fecha (§28). La capa operativa (§1) asigna el esfuerzo en paralelo. **Este plan fija definiciones y contratos; no exige matemática resuelta: todo resultado queda abierto, etiquetado y con gate.**

---

## 1. Capa operativa

**Reparto.** A empuja la cadena teórica (A2 → Lema A → formalización del objetivo → celda dinámico-binaria → frontera). B empuja la cadena de instrumentos (reproducibilidad → tests ancla → interfaz de scorers → oráculo → falsificador → candidatas → atlas). El soporte IA empuja literatura, arbitra borradores y arma harness — subordinado a la validación de A y B. Las tres cadenas son paralelas; se sincronizan en los gates.

**Semana del 3-ago (asignación de esfuerzo, no fechas de teoremas):**

| Día | A | B | Soporte IA |
|---|---|---|---|
| 1 | A-M0–A-M3: modelo + ejercicios ancla + retomar A2 | B-M0–B-M4: auditoría de APIs, tests ancla (**G1 a ciegas**), reproducibilidad | literatura (C-M1) |
| 2 | A-M5–A-M7: núcleo algebraico del Lema A | B-M5–B-M6: interfaz de scorers (cableada con $S_0$) + oráculo de rollout (**G5**) | — |
| 3 | A-M10–A-M12: $S_0$ formal, obstrucciones de S2, diseño de $\varphi_{\mathrm{virgin}}$ y $\varphi$ | B-M7–B-M8: harness del acid test + falsificador | árbitro de A-M12 |
| 4 | A-M13: **G4a — acid test analítico, en papel** | Si G4a aprueba → B-M9: candidatas $S_3$; si no → iterar diseño con A | harness de garantía |
| 5 | A-M15: celda dinámico-binaria (primer intento) + guion de sesión | B-M10: **G4b — acid test computacional** + rankings vs. oráculo + revisión cruzada | lupa del paquete |

La fusión sigue viva: la columna del surrogate en el atlas es a la vez acid test en malla y campo del falsificador; la Conjetura C (P21-A8, fechada con Francisco) se reformula **a la vista de** esa columna. Atlas y garantía solo después de G4b. El calendario cede ante los gates, nunca al revés.

**Gates bloqueantes:** G0 (modelo), G4a/G4b (acid test), G5 (oráculo), G9 (escala). Los criterios de cierre por entregable — **los gates G0–G10: once etiquetas y doce checkpoints efectivos, al dividir G4 en G4a y G4b** — están en §25.

**Prioridad operativa:** durante E1–E3 domina la cadena modelo e inferencia laminar → scorer → acid test → rollout → falsificación/garantía. La frontera $B{=}2/B{=}3$, el atlas extendido, los certificados y la escala no compiten por esfuerzo hasta cerrar esa cadena. Única excepción declarada: la línea (b) de §0-bis (vía submodular), que corre en paralelo vía A-M17 y literatura (C-M1) sin bloquear la cadena.

---

## 2. Interpretación de la dirección

**2.1 El objeto.** $V^* = \max_{\pi\in\Pi^{DA}} \mathbb E[U(\pi)]$; $\pi$ es un árbol de decisión (nodo = historial; acción = pool; rama = conteo; hojas = bienestar). La pregunta: ¿algoritmos eficientes que aproximen el óptimo en ese espacio?

**2.2 La separación es motivación, no destino.** En alta infección el binario agrupado casi siempre es positivo y no dice cuántos; el conteo revela estructura ($R = G{-}1$ ⟹ exactamente una sana); la política subdivide; cobertura + búsqueda acreditan. Mecanismo esencial: **una prueba puede ser valiosa porque crea un estado posterior que vuelve valiosas pruebas futuras.** Por eso la recompensa inmediata no basta.

**2.3 La restricción laminar.** $A \cap B \in \{\varnothing, A, B\}$: permite restar conteos, construir átomos, factorizar la posterior, calcular predictivas exactas y mantener representación cerrada bajo acciones compatibles. Clase de políticas diseñada conjuntamente con su mecanismo de inferencia.

**2.4 La frontera.** Incluso dentro de la clase laminar, el greedy inmediato pierde el mecanismo de separación. Pregunta de diseño: ¿qué objetivo por iteración valora la resolubilidad futura de los átomos *y del territorio virgen*?

**2.5 La garantía, en su orden.** (1) definir la función; (2) verificar que recupera el ejemplo; (3) implementar; (4) comparar contra rollout; (5) buscar contraejemplos; (6) solo entonces, intentar una prueba. Estado actual: para la utilidad terminal hard-clearing con pools libres existe una **derivación condicional de imposibilidad** (§9-C5). La pregunta viva NO se formula como "AS de $\widehat V$": es **determinar si existe una función adaptativa asociada al surrogate — sobre ground set y realizaciones parciales fijos — cuyas ganancias marginales coincidan con el scorer o lo acoten**, o en su defecto un diminishing returns indexado por presupuesto (§20). La garantía natural de $S_3$ puede no ser submodularidad.

---

## 3. Tesis provisional

> Los resultados de conteo pueden crear valor donde los tests binarios agrupados son poco informativos, pero explotar ese valor exige planificación. Las historias laminares ofrecen una clase estructurada donde los conteos se convierten en restricciones sobre átomos disjuntos y las expectativas futuras se calculan exactamente. La evidencia pequeña sugiere que la restricción laminar pierde relativamente poco frente al óptimo irrestricto, mientras que una mala jerarquía y la miopía pierden mucho más. El proyecto diseñará una función de utilidad realizable bajo presupuesto, la comparará contra rollout exacto y determinará qué garantía admite.

**Deslinde de procedencia:** la primera mitad (los conteos crean valor; explotarlo exige planificación; la clase laminar como espacio de búsqueda) es espina del programa (§0-bis a); la apuesta específica — un potencial realizable bajo presupuesto que capture el valor de rollout — es hipótesis propia del equipo (§36).

Cuatro niveles epistemológicos: separación conjunta [resultado bajo hipótesis]; inferencia laminar exacta [implementada y validada; demostración incompleta]; poca pérdida laminar [evidencia finita, no cota]; surrogate planificado [pregunta de investigación].

---

## 4. Pregunta principal y subpreguntas

**Principal:** ¿cuánto bienestar del óptimo dinámico aumentado conserva una política computable que mantiene historia laminar y usa una función local con planificación incorporada?

**Información:** (1) ¿cuándo el conteo supera al binario? (2) ¿qué parte de la mejora es del conteo? (3) ¿cuál de la adaptación? (4) ¿qué resultados intermedios crean más valor futuro? (5) ¿por qué cambia con prevalencia, $G$, $B$, heterogeneidad?

**Estructura:** (6) ¿cuánto cuesta la laminaridad? (7) ¿cuándo el óptimo irrestricto es representable laminarmente? (8) ¿por qué $B{=}1$ es gratis? [respondida: §9-C10] (9) ¿por qué la malla da igualdad en $B{=}2$ homogéneo? (10) ¿qué aparece en $B{=}3$? (11) ¿cuándo un cruce es indispensable?

**Algoritmo:** (12) ¿cuánto cuesta una jerarquía particular? (13) ¿cuánto la miopía? (14) ¿cuánto recupera rollout? (15) ¿puede un surrogate barato reproducirlo? (16) ¿qué parte del tensor necesita? (17) ¿hace falta materializar tablas completas? [implicación de diseño pendiente de validar con perfilado: consultas por demanda — B-M2]

**Garantía:** (18) ¿dentro de jerarquía fija? (19) ¿respecto del óptimo laminar? (20) ¿por régimen? (21) ¿existe una función adaptativa asociada al surrogate, o diminishing returns indexado por presupuesto? (§20) (22) si falla, ¿qué complementariedad la rompe? (23) ¿certificado por instancia?

---

## 5. Modelo normativo (la convención única — gate G0)

**5.1 Individuos.** $Z_i \in \{0,1\}$ ($1$ = activo), $p_i = P(Z_i{=}1)$, $q_i = 1-p_i$, $u_i \ge 0$. Prior producto; independencia solo inicial.

**5.2 Pools.** $T \subseteq [n]$, $1 \le |T| \le G$; el vacío solo como objeto algebraico.

**5.3 Canales.** Aumentado $R(T) = \sum_{i\in T} Z_i$; binario $Y(T) = \mathbf 1\{R(T){>}0\}$. Sin errores.

**5.4 Historial y política.** $H_k$, $b = B-k$, $T_{k+1} = \pi(H_k, b)$; deterministas bastan (MDP finito). **Toda acción cuesta exactamente 1 test**; una variante con otro costo re-declara el modelo.

**5.5 Repetición.** Permitida formalmente; usualmente dominada; podable cuando recompensa y transición son idénticas; no se elimina sin prueba.

**5.6 Acreditados.** $C(H) = \bigcup_{j: R_j = 0} T_j$.

**5.7 Recompensa — strict hard clearing, sin doble conteo:**
$$r(H,T,R) = \mathbf 1\{R{=}0\} \sum_{i\in T\setminus C(H)} u_i, \qquad V^\pi = \mathbb E^\pi\Big[\sum_{k=1}^B r(H_{k-1},T_k,R_k)\Big].$$

**5.8 Regla de la deducción y acreditación parcial.** $D_{\text{healthy}}(H)$ = lógicamente sanos dados los conteos; puede ocurrir $C(H) \subsetneq D_{\text{healthy}}(H)$. Las deducciones informan decisiones pero **no** generan recompensa. La conversión de deducción en utilidad es un problema de **acreditación parcial** mediante pruebas de cero garantizado:

- **Pools libres hasta $G$:** con $b$ pruebas se acreditan hasta $b\cdot G$ individuos deducidos sanos (eligiendo los de mayor utilidad). El presupuesto de acreditación *completa* de un conjunto deducido $D$ es $\kappa_{\mathrm{free}}(H,D) = \lceil |D\setminus C(H)|/G \rceil$.
- **Biblioteca fija $\mathcal T$:** $\kappa_{\mathcal T}(H,D) = \min\{|J| : J \subseteq \mathcal T,\ t \subseteq D_{\mathrm{healthy}}(H)\ \forall t\in J,\ D\setminus C(H) \subseteq \bigcup_{t\in J} t\}$ — la exigencia $t \subseteq D_{\mathrm{healthy}}(H)$ garantiza el cero (y permite que un pool acreditador incluya deducidos sanos de fuera de $D$). Puede ser estrictamente mayor que $\lceil |D\setminus C(H)|/G \rceil$, o $+\infty$ si la cobertura no existe.

$\kappa$ define el costo de acreditación completa, **no** el valor a presupuesto menor: ese es la optimización parcial de §14.6. Sin esta distinción el surrogate valoraría acreditaciones que su propio espacio de acciones no puede ejecutar — parte de separar $V^*$, $V^{*,\mathcal L}$, $V^{*,\mathcal T}$.

**5.9 Variante deductiva.** $r^{ded}$ es un modelo distinto y nombrado; no se mezclan presupuestos, cotas, ejemplos ni teoremas.

**5.10 Masa cero.** Historias factibles; ramas de masa cero excluidas; sin normalizar por cero; motivo registrado.

**5.11 Desempates.** Regla congelada antes de comparar árboles: mayor score → criterio de tamaño declarado → menor máscara.

---

## 6. Definiciones estructurales

**6.1 Familia laminar.** $\mathcal L \subseteq 2^{[n]}$ es laminar $\iff A, B \in \mathcal L \Rightarrow A \cap B \in \{\varnothing, A, B\}$: dos pools son disjuntos o uno contiene al otro; nunca se cruzan parcialmente.

**6.2 Biblioteca laminar fija (ex ante).** Una familia laminar fijada antes de observar resultado alguno; la política puede elegir cualquier acción de la biblioteca en cualquier estado, salvo podas declaradas (§5.5, §14.10).

**6.3 Jerarquía $\mathcal T$.** Una biblioteca laminar con la relación padre–hijo inducida por inclusión: $C$ es hijo de $A$ si $C \subsetneq A$ y no existe $E \in \mathcal T$ con $C \subsetneq E \subsetneq A$. Puede ser un bosque; no se exige que los hijos cubran al padre, ni árbol binario, ni raíz única, ni hojas singleton.

**6.4 Historia laminar.** $H$ es laminar si los pools efectivamente ejecutados $\{T_1, \dots, T_k\}$ forman una familia laminar.

**6.5 Política laminar ex ante.** Restringida a una biblioteca fija; es el objeto del atlas y de $V^{*,\mathcal L}$.

**6.6 Política laminar ex post.** Construye su familia adaptativamente, con la restricción de que la historia ejecutada termine laminar; clase potencialmente más rica; no se confunde con $V^{*,\mathcal L}$. Relación: $V^{*,\mathcal L_{ex\,ante}} \le V^{*,\mathcal L_{ex\,post}} \le V^*$; sobre las mismas instancias, la ex post tiene **razón de desempeño** al menos tan alta y **pérdida relativa** a lo sumo tan grande — razón 0.928 (pérdida 0.072) en malla, 0.9069 (pérdida 0.0931) adversaria; evidencia finita, jamás cota. Normativa actual: ex ante; ex post como extensión.

**6.7 Átomo residual.** $D_A = A \setminus \bigcup_{C \in \operatorname{children}(A)} C$ — la unión recorre **solo los hijos** de $A$ en la jerarquía. Un átomo no es necesariamente una hoja.

**6.8 Territorio virgen.** $V(H) = [n] \setminus \bigcup_{(T,R) \in H} T$; conserva prior producto e independencia respecto de los átomos condicionados.

---

## 7. Valores de referencia y cadenas

$V^*$, $V^{*,\mathcal L}$, $V^{*,\mathcal T}$, $V^{g,\mathcal T}$, $V^{r,\mathcal T}$, $V^{s,\mathcal T}$, $V^{stat,bin}$, $V^{dyn,bin}$. Cadenas: $V^{stat,bin} \le V^{dyn,bin} \le V^{dyn,count}$; $V^{*,\mathcal T} \le V^{*,\mathcal L} \le V^*$; $V^{g,\mathcal T} \le V^{r,\mathcal T} \le V^{*,\mathcal T}$ (la primera bajo las hipótesis de la Proposición B). Todas las cantidades comparadas usan el mismo modelo, presupuesto, convención, espacio de estados y evaluaciones exactas o declaradamente comparables. **No se asume** $V^{g,\mathcal T} \le V^{s,\mathcal T}$: se verifica o se demuestra.

---

## 8. Descomposición de pérdidas

$$\frac{V^{s,\mathcal T}}{V^*} = \rho_{\text{plan}}\cdot\rho_{\text{tree}}\cdot\rho_{\text{lam}}, \qquad \rho_{\text{plan}} = \frac{V^{s,\mathcal T}}{V^{*,\mathcal T}},\ \rho_{\text{tree}} = \frac{V^{*,\mathcal T}}{V^{*,\mathcal L}},\ \rho_{\text{lam}} = \frac{V^{*,\mathcal L}}{V^*}$$

con diferencias absolutas $\Delta$ reportadas siempre. Denominador cero ⟹ `NaN` + diferencia absoluta + etiqueta; jamás razón imputada 1. Un greedy laminar malo no demuestra que la clase laminar sea mala.

---

## 9. Registro de claims corregidos y lenguaje permitido

Registro vivo; números verificados contra CSV el 1-ago-2026. Toda cifra citable sale de aquí.

| # | Claim viejo | Lenguaje permitido |
|---|---|---|
| C1 | "Greedy laminar domina al estático casi siempre" | Malla (2,592 instancias): greedy gana **67.0%** global; **96.0%** en prevalencia alta (**rollout: 98.2%**); **40.4%** en baja (`showcase_regions.csv`). Dominancia de régimen. |
| C2 | "Producto de marginales exacto para descendientes compatibles" | Factorización **entre átomos**, no entre individuos; dentro, Bernoulli condicional. |
| C3 | "La caché acelera ~98,000×" | Para $G{=}10$, la reutilización **evita una mediana de 97,274 convoluciones** (medianas $G{=}4/6/8/10/12$: 100 / 1,012 / 11,076 / 97,274 / 174,076, `subset_tables.csv`); tras la caché, cero nuevas. **No es un speedup** (sin razón finita con denominador cero): pared 1.3–1.8×, materialización y overhead dominan. Una razón de complejidad exigiría denominador no nulo (p. ej., costo amortizado tras $K$ consultas). |
| C4 | "La separación aísla el valor del conteo" | Separación **conjunta**; la celda dinámico-binaria está pendiente (§18). |
| C5 | "Greedy falla ⟹ no hay AS" | **[DERIVACIÓN CONDICIONAL — formalización pendiente → A-M17]** Bajo el mapeo pools-como-items con presupuesto cardinal (ground set: pools admisibles, cada uno una vez; realización parcial: conteos observados; utilidad terminal: unión de individuos en pools con cero observado), la ganancia marginal esperada es exactamente $S_0$. En el ancla, $V^{S_0} = 0.35u$ (baseline singleton) y una política factible logra $\ge 0.806u$, luego $V^{S_0}/V^* \le 0.434 < 1-1/e$. Verificadas las hipótesis restantes de Golovin–Krause, la utilidad hard-clearing **no es** adaptativamente submodular bajo ese mapeo; tras A-M17 sube a [DEMOSTRADO]. El falsificador busca además el testigo directo $(\psi,\psi',t)$. **No murió** para: jerarquías fijas, otros ground sets, otras restricciones, versiones aproximadas — la pregunta viva sobre el surrogate es la de §20. |
| C6 | "Biblioteca laminar = matroide laminar" | No establecido; no se invoca sin axiomas. |
| C7 | "El Gibbs está resuelto" | Estacionariedad validada en topologías auditadas. El barrido de irreducibilidad existe como commit histórico (`6eee18e`, contiene `augmented/irreducibility_sweep.py`) **fuera de la rama de trabajo**: pista recuperable, no evidencia activa — no se cita hasta recuperar, reintegrar, re-ejecutar y registrar. Mixing abierto. Apéndice; no compite con A2/Lema A/surrogate. |
| C8 | "Peor caso laminar de malla: 0.943" | Atlas completo: razón **0.928** malla, **0.9069** adversaria (evidencia finita, no cota). |
| C9 | "Ley del lookahead 99/40/16" | Retirada (artefacto de cableado). Permitido: lookahead de un paso bien cableado recupera ~90% del hueco a todo horizonte medido. |
| C10 | **"La primera oportunidad de adaptación"** | *Estructural:* con $B{=}1$ no puede existir adaptación (una sola decisión). *Empírico:* 210/210 instancias homogéneas con $B{=}1$ dan $V^{*,\mathcal L} = V^*$ (`homogeneous_b2.csv`) — la restricción ex ante es gratuita con una acción. *Abierto:* en qué instancias el valor adaptativo es estrictamente positivo desde $B{=}2$; oportunidad ≠ mejora estricta. |

---

## 10. Auditoría del plan del 21 de julio (tarea → estado → milestone / gate / responsable)

| ID | Ítem | Estado | Milestone / Gate / Resp. |
|---|---|---|---|
| P21-B1 | Inferencia laminar | Terminada en código, validada | Prueba formal → §13 / G2 / A; API → B-M0 / — / B |
| P21-B2 | Scenario MILP | Terminado | Muestra vs. población → §30.3 / — / B |
| P21-B3 | Proposición B | Sustancialmente terminada | Notación §5; código = política demostrada → B-M6 / G5 / B |
| P21-E1 | Cinco cantidades | Terminado | — |
| P21-E2 | Atlas v1 | Terminado (2,592, trazable) | Extensión tras G4b/G6 → B-M13 / G8 / B |
| P21-E3 | Adversaria | Evidencia (0.928/0.9069) | Siembra §19 → B-M11 / — / B |
| P21-E4 | $B\le2$ | B=1 estructural; B=2 igualdad ~1e-15 | Teorema → §19 / G7 / A |
| P21-E5 | Gap de independencia | Terminado | Lenguaje C2 |
| P21-E6 | MILP por partículas | Terminado | — |
| P21-E7 | Pipeline n=40 | Prueba arquitectónica | Alimenta R6 |
| P21-A1–A6 | Lema A completo | (i) casi; (ii) parcial; resto pendiente | §13 / G2 / A (A-M5–A-M9) |
| P21-A7 | Peor caso | Parcial | Patrón estructural → A-M14 / G6 / A |
| P21-A8 | Conjetura por régimen | Reformulación (C1) | Tras columna surrogate → §23.4 / G8 / A |
| P21-A9 | Paquete | Sustituido | §34 / G10 / A+B |
| P21-S | Treewidth / figura | Futuro / parcial | No antes del algoritmo |

---

## 11. Auditoría del plan del 27 de julio

| ID | Ítem | Estado | Milestone / Gate / Resp. |
|---|---|---|---|
| T0 | Higiene | Parcial (`9737a99`) | `.gitignore`; fuera del eje |
| T1–T2 | Oráculo + forma cerrada + caché | **Terminados, reubicados en `laminar_tables.py`** | — |
| T3 | Checks de Francisco | Terminados | — |
| T4 | Tests ancla | Parcial | Hipergeométrico, 11/10/5, $n{=}3$, gate a ciegas → B-M1 / G1 / B+A |
| T5 | Demo | Ausente | Solo pre-sesión si aporta más que notebook 22 |
| T6 | Falsificador | Ausente | §17 → B-M8 / G6 / B |
| T7 | Literatura | Ausente | §21 → C-M1 / — / soporte IA, valida A |
| T8 | Congelación | No ejecutada | Reproducibilidad → B-M4 / — / B |
| A1 | Ejercicios manuales | No documentados | A-M1, A-M2 / G1 / A |
| A2–A6 | Nota tensor + lupa + final + guion + revisión | A2 iniciado, archivo ausente | §12 → A-M3–A-M4, A-M20 / G1 / A |
| A7–A9 | Lema A(iii), coherencia, literatura | Pendientes | §13, §21 / G2 / A |
| A10–A11 | Análisis falsificador, paquete | Bloqueado / absorbido | A-M14, §34 |
| B7–B8 | Revisión módulo + pools adversarios | Parcial de facto | Auditoría + cobertura → B-M0, B-M1 / — / B |
| B9 | Wrapper por átomo | No obligatorio | **YAGNI:** solo si el scorer lo pide |
| B10–B11 | Greedy miope como meta / validación | Reemplazados | Scorer planificado; validación → §24 |
| — | Tablas incrementales | **Terminadas** (el plan las recortaba) | Costo interpretado en C3 |

---

## 12. Nota técnica del tensor — A2

**Objetivo:** el tensor como objeto matemático, implementado, validado y conectado con la política. **Archivo:** `augmented/paper/nota_tensor_subpruebas.md` (3–4 pp.).

- **A2.1 Definiciones.** $Q_{T,R}(S,r) = P(R(S){=}r\mid R(T){=}R)$, $S \subseteq T$; $m$, $s$, $f_A(k)$ Poisson-binomial prior; coordenadas locales.
- **A2.2 Soporte.** $\max(0, R-(m-s)) \le r \le \min(R,s)$; dos renglones por el complemento.
- **A2.3 Forma cerrada.** $Q = f_S(r) f_{T\setminus S}(R-r)/f_T(R)$ (Bayes; independencia prior entre bloques).
- **A2.4 Homogéneo.** Hipergeométrica $\binom{s}{r}\binom{m-s}{R-r}/\binom{m}{R}$; $p$ se cancela.
- **A2.5 Extremos.** $R{=}0$; $R{=}m$; $S{=}\varnothing$; $S{=}T$.
- **A2.6 Fila cero.** $Q(S,0) = f_S(0) f_{T\setminus S}(R)/f_T(R)$; score exacto = score de independencia × $f_{T\setminus S}(R)/f_T(R)$ — el gap de independencia en un número.
- **A2.7 Marginales y dependencia.** $P(Z_i{=}1\mid R) = p_i f_{T\setminus\{i\}}(R{-}1)/f_T(R)$; la conjunta no factoriza por individuo. **[CONJETURA RESPALDADA EMPÍRICAMENTE 0/2000; LITERATURA PENDIENTE]** el producto de marginales sobreestimó $P(\text{limpio})$ en todas las instancias verificadas; queda por comprobar que Joag-Dev–Proschan (1983) aplica a Bernoulli **heterogéneas** condicionadas a su suma e implica la desigualdad del evento conjunto de ceros (→ §21).
- **A2.8–A2.10 Caché, split, herencia.** $\Phi_T = \{f_S\}$ por DP; split ⟹ átomos $(S,r)$, $(T\setminus S, R-r)$; los hijos reutilizan $\Phi_T$.
- **A2.11–A2.12 Costos y evidencia.** Separar construcción / lookup / convoluciones / materialización / memoria / pared. Evidencia: texto de C3 (medianas; sin razones con denominador cero); configuración, candidatos, memoria, tiempo, hardware registrados.
- **A2.13 API.** `subset_pmf_cache`, `subpool_tensor`, `subpool_tensor_brute`, `split_after_test` (en `laminar_tables.py`).
- **A2.14–A2.16 Relaciones y límites.** Caso local del Lema A ($L{=}\{T\}$); habilita recompensa inmediata, ramas, imposibles, potencial, rollout local; **no decide la acción** — no resuelve jerarquía, planificación, garantía ni clausura tras cruces; costo exponencial en $G$ si se materializa todo.

**Tests ancla:** $n{=}3$, $p=(\tfrac12,\tfrac12,\tfrac12)$, $R{=}2$ (110/101/011 uniformes); $11/10/5$ (soporte $\{4,5\}$, $5/11$, $6/11$; heurística-producto con masa fuera del soporte). **Cierre:** prueba y código coinciden; A entrega ejemplos a ciegas (G1); B revisa claims; cero costos ambiguos.

---

## 13. Lema A completo

**Hipótesis:** familia laminar finita; jerarquía válida; prior producto; conteos coherentes; pools no vacíos; masa positiva.

**Parte I — Partición.** $\{D_A \neq \varnothing\}$ particiona $\bigcup_A A$; sublemas: hijos disjuntos, átomo⊥hijos, comparables e incomparables disjuntos, cobertura, unicidad.

**Parte II — Conteos.** $c(D_A) = c(A) - \sum_C c(C)$; $c(A) = \sum_{B\in\text{subárbol}} c(D_B)$; equivalencia pools ⟺ átomos (inducción).

**Parte III — Factorización.** $P(Z{=}z\mid E_H) = \prod_D P(Z_D{=}z_D\mid R(D){=}c_D)\cdot P(Z_V{=}z_V)$, normalización demostrada; dependencia interna no factoriza.

**Coherencia.** $0 \le c_D \le |D|$; $\sum_C c(C) \le c(A) \le |D_A| + \sum_C c(C)$; duplicados con mismo conteo; jerarquía = inclusión; masa positiva. Los `ValueError` del código son estas hipótesis.

**Corolarios.** Marginal $p_i f_{D\setminus\{i\}}(c{-}1)/f_D(c)$; ley predictiva por convolución sobre $t = (t\cap V)\cup\bigcup_D(t\cap D)$ — incluso cruzados; **evaluar ≠ ejecutar** (la PMF de una acción cruzada se calcula una vez; la posterior resultante puede dejar de factorizar); complejidad por etapas, sin $O(\cdot)$ mezclada.

**Tests 1:1:** cruce, duplicado, fuera de rango, padre–hijo imposible, jerarquía incorrecta, masa cero, historia vacía, ramificada, pool cruzado futuro, virgen, unión de átomos, descendiente.

Ejercicio ancla: $n{=}4$, dos átomos, a mano. Fallback: dos bloques disjuntos. Destino: §7.3 del documento compartido (trampas: $A$ = matriz de incidencia — usar $s,t$; $Z$ = normalizador, indicador $X_i$; población $[n]$).

---

## 14. Diseño de la función objetivo

**Linaje:** la conversación registra la intuición de valorar los sanos esperados si las pruebas posteriores fueran gratuitas. El equipo la formaliza mediante $S_2$; las obstrucciones de 14.4 motivan $S_3$, basado en utilidad realizable bajo el presupuesto residual. $S_3$ no es una invención desconectada sino la corrección motivada de la idea discutida. La sesión 2026-08-11 añade la variante candidata **valor/costo**: $V(T)$ acompañado de $C(T)$ = costo esperado en pruebas del greedy de extracción (Monte Carlo sobre el posterior), con poda por presupuesto restante — misma moraleja, forma operativa distinta; se evalúa como candidata $S_3$ bajo el mismo régimen G4a/G4b.

**14.1 Requisitos.** $S(H,t,b)$: historial completo; hard clearing; descuenta lo acreditado; respeta presupuesto; valora resolubilidad del virgen y de los átomos; sin doble conteo (en particular $r$/$U(C)$, ver 14.5); reproducible; más barata que el óptimo; comparada contra rollout.

**14.2 $S_0$.** $S_0(H,t) = P(R(t){=}0\mid H)\sum_{i\in t\setminus C(H)} u_i$. Con $q=\varepsilon$ nunca agrupa.

**14.3 $S_1$ colapsa.** Hard clearing no acredita deducciones ⟹ $S_1^{hard} = S_0$ idénticamente. ($S_1^{ded}$: modelo deductivo, §5.9.)

**14.4 $S_2$ — dos variantes, dos obstrucciones [DERIVACIÓN — pruebas completas en A-M11].** **Global** $\Phi_2 = \sum_{i\in[n]} u_i P(Z_i{=}0\mid H)$: muere por tower property (incremento esperado cero ante toda acción) [A-M11a]. **Cubierta** $\Phi_2^{cov} = \sum_D\sum_{i\in D} u_i P(Z_i{=}0\mid c_D)$: un pool virgen la incrementa en $\sum_{i\in t} u_i q_i$ (recupera el primer movimiento), pero es martingala bajo subdivisión: jamás extrae [A-M11b]. **Moraleja:** el potencial debe ser **realizable bajo presupuesto**.

**14.5 $S_3$ — potencial de continuación incremental.**

$$\Phi_b(H) = \max_{\substack{b_0 + \sum_{D\in\mathcal A(H)} b_D \le b\\ b_0, b_D \in \mathbb Z_{\ge0}}}\Big[\varphi_{\mathrm{virgin}}(V(H), b_0) + \sum_{D\in\mathcal A(H)} \varphi(D, c_D, b_D)\Big], \qquad Q_{\Phi,b}(H,t) = \mathbb E\big[r(H,t,R_t) + \lambda_b \Phi_{b-1}(H') \mid H,t\big]$$

**Sin doble conteo:** el potencial no incluye $U(C(H))$; sumar $r + \widehat V_{b-1}$ con $\widehat V = U(C)+\Phi$ contaría $r$ dos veces ($U(C(H')) = U(C(H)) + r$). $\widehat V_b = U(C(H)) + \Phi_b(H)$ existe solo para reporte, nunca en el scorer. Casos base: $\Phi_0 = 0$; consistencia $Q_{\Phi,1} = S_0$. Restar $\Phi_b(H)$ no cambia el $\arg\max$ (constante en $t$).

**Separabilidad como aproximación falsificable:** la factorización posterior justifica calcular contribuciones probabilísticas por bloque, pero **no demuestra separabilidad del problema de control**: acciones multiátomo, mezclas átomo–virgen y restricciones conjuntas de la biblioteca pueden crear sinergias que la forma knapsack omite. Es una aproximación estructurada; su realizabilidad se verifica por candidata y el falsificador mide específicamente el regret de estas interacciones (§17).

**$\lambda_b$ (candado anti-overfitting):** $\lambda_b = 1$ para el candidato principal; descuento solo con razón estructural; si se calibra: diseño/validación/holdout adversarial, congelado antes de evaluar; **jamás ajustar y reportar sobre el mismo atlas**.

**14.6 $\varphi$ con estado local enriquecido y acreditación parcial.** $\varphi(D,c,b)$ es **abreviatura** de $\varphi_{\mathcal X}(\xi_D(H), b)$, con $\xi_D(H) = (D, c_D, C(H)\cap D, \mathcal X, \text{contexto deducido})$ y $\mathcal X$ el espacio de acciones evaluado — dos historias con el mismo triple pueden tener distinto valor acreditable. Requisitos: $\varphi(\xi,0) = 0$; $\varphi = 0$ si $c = |D|$; monótona en $b$ donde corresponda; acotada por la utilidad sana posible; computable; compatible con split; sin doble conteo; explícita sobre acreditación. Caso $c = 0$ — el criterio es **pertenencia a $C(H)$**, no el origen del cero; el valor es la **acreditación parcial**:

$$\varphi_{\mathrm{free}}(D, 0, b) = \max_{\substack{S \subseteq D\setminus C(H)\\ |S| \le bG}} U(S) \quad \big(= u\min\{|D\setminus C(H)|, bG\} \text{ si } u \text{ homogénea}\big),$$

$$\varphi_{\mathcal T}(D, 0, b) = \max_{\substack{J \subseteq \mathcal T,\ |J| \le b\\ t \subseteq D_{\mathrm{healthy}}(H)\ \forall t \in J}} U\Big(\big[D \cap \bigcup_{t\in J} t\big] \setminus C(H)\Big).$$

$U(D)\cdot\mathbf 1\{b \ge \kappa(H,D)\}$ sobrevive solo como caso especial "acreditación completa".

**14.7 $\varphi_{\mathrm{virgin}}$ — política restringida computable, no un Bellman escondido.** Candidato principal $\varphi_{\mathrm{virgin}}^{\mathrm{CBS}}(V, b_0)$: valor esperado de una política canónica *cover–binary-search* con todas las reglas fijas (pools raíz disjuntos de tamaños predeterminados; cobertura fija; selección del primer pool con conteo no extremo; subdivisión balanceada; test final de acreditación; abandono si el residuo no completa la ruta). Declara familia, reglas, costo, y que es **cota inferior realizable**. $\varphi_{\mathrm{virgin}}(\varnothing, b_0) = \varphi_{\mathrm{virgin}}(V, 0) = 0$. En la familia del acid test, CBS es el plan $k$ pools + búsqueda binaria: el potencial ve el primer movimiento por construcción.

**14.8 Taxonomía de $\varphi$** (cada candidata declara: información / complejidad / cota inf–sup–heurística / **realizable sí-no** / **acción factible bajo la misma biblioteca sí-no** / casos base / acid test / doble conteo / hard clearing / conexión con garantía): **A.** acreditación directa (con $\kappa_{\mathrm{free}}$ o $\kappa_{\mathcal T}$ y las fórmulas parciales de 14.6); **B.** resolubilidad con $b$ niveles; **C.** DP local por átomo; **D.** relajación optimista — cota superior, **no compite como política sin calibración**; **E.** aproximación aprendida del rollout — solo diagnóstico. La lectura de sesión 2026-08-02 — "valor extraíble por componente + explorar la rama de valor máximo" [22:02–23:50] — es la instancia informal de las candidatas A–C sobre $\varphi$(átomos); cada candidata declara cómo la refleja.

**14.9 Casos base exactos.** $c = |D|$: $\varphi = 0$. $c = 0$: acreditación parcial de 14.6 (los ya en $C(H)$ valen 0). Caso $c_D = |D|-1$ — **costo de la política canónica, no propiedad universal del estado**: si $D\cap C(H)=\varnothing$ y $c_D=|D|-1$, la política canónica localiza al único sano en a lo sumo $\lceil\log_2|D|\rceil$ pruebas de subdivisión bajo la regla declarada. Algunas ramas terminan con un singleton cuyo cero fue observado; otras identifican al sano por descarte y requieren una prueba adicional para acreditarlo. Se reportan por separado costo esperado y peor caso.

**14.10 Selección.** Candidato ∈ espacio declarado; descartar lo acreditado; podar dominadas; desempate fijo (§5.11).

---

## 15. Rollout oracle

Base: greedy $S_0$ exacto. $Q_b^g(H,t) = \mathbb E[r + V_{b-1}^g(H')\mid H,t]$ con $V^g$ **incremental** (misma convención que 14.5); $\pi^r_b = \arg\max_t Q^g_b$, replanificado en cada estado. Estado completo: mundos exactos, acreditados, presupuesto, átomos, biblioteca. Validación: DP sobre ramas vs. simulación por perfil latente, $10^{-10}$; punto de partida `rollout_laminar_value`/`ExactPolicyEvaluator` (P21-B3 verifica código = política demostrada). Límites tras perfilado. Métricas: top-1, top-k, Kendall/Spearman, regret local, bienestar, tiempo, memoria, consultas.

---

## 16. Acid test

**Familia:** prior homogéneo, $q\ll1$, $u$ homogénea, pools raíz de tamaño $G$, población suficiente, strict hard clearing. **Ancla ejecutable (audit):** $(q,G,k,B) = (0.05,16,2,7)$ — dinámico-conteo $\ge 0.806u$ contra $0.35u$ = **valor del baseline singleton $S_0$ (coincide con el óptimo estático en esta ancla)** — misma nomenclatura que C5. Forma general strict-clearing: $k = \max\{0,\ B - \lceil\log_2 G\rceil - 1\}$ pools raíz (la expresión sin techo vale exactamente cuando $G$ es potencia de dos, como en todo el barrido declarado $G \in \{2,4,8,16\}$), cota $u[1-(1-q)^{kG}]$ con el test acreditador contado. Cambios de convención pasan por G0.

**G4a — analítico (bloqueante, en papel):** la forma candidata asigna valor al primer movimiento y no contradice los casos base de 14.9. **G4b — computacional:** candidato implementado; **nueve checks de trayectoria completa:** (1) valora abrir territorio virgen; (2) tras conteo informativo, valora volver al pool útil; (3) subdivide con presupuesto suficiente; (4) distingue cero observado de limpieza deducida; (5) reserva el test acreditador; (6) no gasta todo explorando; (7) no duplica utilidad acreditada; (8) robusto al variar $q,G,k,B$; (9) no depende de un desempate afortunado.

**Vecindad:** varios $q<1/2$; $G\in\{2,4,8,16\}$; $k\in\{1,2,3\}$; perturbaciones de $u$ y priors. **Criterio inicial:** mecanismo correcto + bienestar $\ge S_0$ en la familia + ventaja en rango no trivial + regret razonable; umbral cuantitativo tras pilotos. Resultados permitidos: recupera / en sub-rango documentado / no recupera. Prohibido ajustar ad hoc sin re-correr el falsificador (R2).

---

## 17. Falsificador de comportamiento de políticas

**Políticas:** $S_0$, cada $S_3$, rollout, óptimo de $\mathcal T$, óptimo laminar, óptimo irrestricto ($n\le6$). **Estados:** factibles, masa positiva, alcanzables; banco común para comparar scorers fuera de sus trayectorias.

**Cruce, formal:** $t$ cruza $T_j \iff t\cap T_j \neq \varnothing,\ t\setminus T_j \neq \varnothing,\ T_j\setminus t \neq \varnothing$ (equiv. $t\cap T_j\notin\{\varnothing,t,T_j\}$); cruzada respecto de $H$ ⟺ cruza algún pool ejecutado. Fuera: disjuntas, descendientes, ancestros, repeticiones. Se distinguen: compatible con la historia realizada / permitida ex ante / laminar solo en algunas ramas.

**Clases (primaria por precedencia + flags):** repetida / descendiente / ancestro / unión de átomos / compatible mixta / virgen / cruzada / dominada. **Flags de separabilidad:** intraátomo / **multiátomo** / **átomo–virgen** / **valor perdido por separabilidad** / política local realizable individualmente pero no conjuntamente. Si $S_3$ falla por acciones multiátomo, se habrá identificado exactamente qué complementariedad falta — resultado, no fracaso.

**Ponderación:** $W_c^\pi = \sum_H P^\pi(H)\mathbf 1\{\text{class}=c\}$. **Métricas:** fracción por clase, primer anidamiento, profundidad, tamaños, entropía, inmediata, potencial, $Q$-rollout, regret, bienestar. **Barrido:** $n\in\{4,5,6\}$, $B\in\{1,2,3\}$, $G\in\{2,3\}$, prevalencia 0.05–0.90, priors y utilidades homo/heterogéneas. **Salidas:** CSV por decisión (instance_id, history_id, probability, policy, action, class, flags, score, immediate, rollout_q, local_regret, final_value) y por instancia. La curva se reporta tal como salga.

---

## 18. Celda dinámico-binaria (identificadora, sin resultado predicho)

$V^{dyn,bin}$ con $Y = \mathbf 1\{R>0\}$:

| Política | Observación | Adaptación |
|---|---|---|
| Estática binaria | binaria | no |
| Dinámica binaria | binaria | sí |
| Dinámica count | conteo | sí |

Desenlaces posibles sin apostar: casi todo adaptación / casi todo conteo / complementariedad / por régimen. Contexto (no predicción): el paper hermano prueba que con $q<1/2$ el pooling binario no paga *en la celda estática*. Empírico: DP exacto, simetría, políticas extraídas. Teórico: reemplazo pool→singleton *considerando continuación*; vigilar: rama cero de pool grande, positivas que cambian asignación, agotamiento finito, deducciones por traslape. Fallback: cota + limitación + separación conjunta (C4).

---

## 19. Frontera $B{=}2/B{=}3$

**Hipótesis a decidir** (el programa no favorece ningún desenlace): bajo $p_i = p$, $u_i = u$, $|T|\le G$, hard clearing, población finita, ¿$V^{*,\mathcal L} = V^*$ para $B{=}2$? (Evidencia: igualdad ~1e-15 en malla, `homogeneous_b2.csv`.) Programa: catálogo de pares (disjuntos/anidados/cruce); reemplazo laminar; simetría $(|A|,|B|,|A\cap B|,R_1)$; enumeración por tamaños; casos frontera ($G{=}1$, $G{=}n$, $R{=}0$, $R{=}|T|$); contraejemplo heterogéneo; menor contraejemplo en $B{=}3$ (siembra B-M11) e identificación de qué lo produce.

---

## 20. La pregunta de garantía (cinco preguntas separadas)

Formalización obligatoria antes de programar: ground set, items, realización, observación, realización parcial, dominio, orden, factibilidad, utilidad. Las cinco preguntas, por separado:

1. **AS estándar de la utilidad terminal hard-clearing** — estado: derivación condicional de imposibilidad bajo pools-como-items (C5); formalización A-M17.
2. **AS estándar de una utilidad relajada fija** (p. ej., cobertura relajada) — abierta.
3. **Diminishing returns condicional del potencial $\Phi_b$** — $\Phi_b$ depende del presupuesto y contiene una optimización interna: **no es** una función adaptativa estándar; se estudia indexado por presupuesto o con estado aumentado, y **no se denomina adaptive submodularity sin construir la reducción** (una función adaptativa asociada cuyas ganancias marginales coincidan con el scorer o lo acoten).
4. **Policy improvement del scorer:** ¿$V^{S_3} \ge V^{S_0}$? — a verificar o demostrar (no se asume, §7).
5. **Aproximación respecto de rollout:** cotas de regret local.

Falsificador: buscar $\Delta(t\mid\psi') > \Delta(t\mid\psi) + \varepsilon$ con aritmética exacta donde sea viable, testigo completo, doble implementación; violación = complementariedad informativa (resultado, no fracaso). Mapeo prioritario: nodos de jerarquía fija. Matroides: C6. **Nota honesta:** la garantía natural de $S_3$ puede no ser submodularidad — puede ser policy improvement, regret acotado, garantía por régimen, o una relación entre $\Phi_b$ y el verdadero valor de continuación.

**Seguimiento de la sugerencia submodular (entrada distintiva de la conversación; interés prioritario de A):** las preguntas 1–3, junto con las relajaciones convexas del barrido de §21, responden directamente a la vía sugerida. Compromiso: el proyecto entregará siempre una respuesta documentada — propiedad o reducción demostrada, limitación caracterizada o contraejemplo explicado — nunca silencio ni lenguaje aspiracional. **Secuencia interna:** la pregunta 1 (A-M17/C5) avanza de inmediato — no depende de $S_3$; la reducción de $\Phi_b$ (pregunta 3) espera a la definición estable de $S_3$ (tras G3/G4a). La elevación de esta línea compromete una respuesta documentada, no trabajo prematuro sobre un objeto sin congelar.

---

## 21. Literatura (soporte IA ejecuta; A valida fuentes primarias antes de incorporar)

Preguntas: ¿el tensor es conditional Bernoulli conocido? ¿la factorización es folclor? ¿QGT adaptativo laminar? ¿binary splitting para conteos? ¿welfare heterogéneo? ¿garantías AS en testing? ¿qué asociación negativa aplica exactamente (A2.7)? ¿value of information y rollout? ¿qué ofrecen las relajaciones convexas de funciones submodulares (extensión de Lovász, relajación multilineal) para el objetivo o su análisis? Categorías: conditional Bernoulli (Chen–Liu 1997), rejective sampling, Poisson-binomial, hipergeométrica, negative association (Joag-Dev–Proschan 1983), group testing, QGT, nested testing, adaptive diagnosis, stochastic optimization, adaptive submodularity (Golovin–Krause 2011+), approximate DP, information relaxation (Brown–Smith–Sun), relajaciones convexas de submodulares (Lovász; Calinescu–Chekuri–Pál–Vondrák; prioridad temprana en C-M1 por §0-bis b), estructuras laminares. Producto: tabla claim/fuente/hipótesis/coincidencia/diferencia/acción en `docs/notes/2026-08-XX-revision-QGT.md`. Reglas: fuentes primarias; no citar resúmenes como prueba; verificar teoremas; sin claims de novedad antes del barrido.

---

## 22. Reproducibilidad

**Meta:** checkout limpio + `pip install -r requirements.txt` + `pytest` = suite base verde, opcionales con skip razonado, sin fallo de colección. **Estado verificado (1-ago):** `gymnasium` (importado por `tests_rl_fixes.py`) no está en requirements y rompe la colección; 5 fallos MOSEK por licencia vencida en `tests_solvers.py`; cero discrepancias numéricas. **Clases de dependencias:** base (numpy, scipy, pandas, matplotlib) / visualización (graphviz, seaborn) / RL (gymnasium, …) / solvers (MOSEK, Gurobi, SciPy MILP) / desarrollo (pytest). **Acciones:** requirements separados o extras; markers `skipif` con razón — **sin caída silenciosa a heurístico cuando el test valida el solver comercial**; tests de fallback separados; CI base determinista + suite opcional. **Semillas y artefactos:** generador, seed, versión, parámetros, commit; CSV canónico por experimento, no solo notebooks.

---

## 23. Matriz experimental

**23.1 Exacta pequeña:** $n\in\{4,5,6\}$, $B\in\{1,2,3\}$, $G\in\{2,3\}$, prevalencia 0.05–0.90, priors homo/beta-bimodal, utilidad plana/heterogénea; políticas $S_0$, $S_3$, rollout, $V^{*,\mathcal T}$, $V^{*,\mathcal L}$, $V^*$. **23.2 Acid test:** $G\in\{2,4,8,16\}$, varios $q<0.5$, $k\in\{1,2,3\}$, utilidad perturbada, presupuesto corregido. **23.3 Falsificador de garantía:** $n\le6$, exhaustivo. **23.4 Atlas extendido — solo tras G4b/G6:** columnas $V^{*,\mathcal T}$, $V^{S_0,\mathcal T}$, $V^{S_3,\mathcal T}$, $V^{r,\mathcal T}$, razones, diferencias, tiempos; aquí se reformula la Conjetura C (P21-A8). **23.5 Escala — solo tras G9:** $n = 20$–$50$, certificados e intervalos, backend visible.

---

## 24. Métricas

Bienestar (media, diferencia, razón, mediana, mín/máx, SE); recuperación del gap $(V^{S_3}-V^{S_0})/(V^{r}-V^{S_0})$ solo con denominador positivo; regret local $Q^{r}(t^*)-Q^{r}(\hat t)$; ranking (top-1, top-k, Kendall, Spearman); costo (tiempo por decisión/episodio, memoria, PMFs, convoluciones, ramas, candidatos); estructura (clases — incluidos los flags de separabilidad —, tamaño, profundidad, átomos).

---

## 25. Gates y disciplina de claims

**Etiquetas obligatorias:** [DEMOSTRADO] / [VERIFICADO n≤X] / [DERIVACIÓN CONDICIONAL] / [CONJETURA RESPALDADA] / [PREGUNTA] — nunca mezcladas; sin test ni prueba, degradación explícita.

**Los gates G0–G10 (once etiquetas y doce checkpoints efectivos, al dividir G4 en G4a y G4b):** **G0** modelo aprobado, sin convenciones conviviendo · **G1** tensor: dos vías a $10^{-12}$, ejemplos a ciegas, claims mapeados · **G2** Lema A: prueba completa, test por hipótesis, revisión cruzada · **G3** objetivo: fórmula cerrada, casos base, complejidad, no circular, **sin doble conteo $r$/$U(C)$** · **G4a/G4b** §16 · **G5** rollout: dos evaluadores a $10^{-10}$ · **G6** falsificación: banco completo, contraejemplos guardados, candidato seleccionado o descartado · **G7** garantía: teorema, prueba o certificado, sin lenguaje aspiracional · **G8** atlas: candidato congelado, outputs canónicos · **G9** escala: entendido en pequeño, estimadores auditados, intervalos · **G10** Francisco: cifras trazables, claims etiquetados, preguntas concretas; el cierre incluye el despacho post-sesión de §34-bis.

**Reglas permanentes:** dos vías independientes; mapeo 1:1 enunciado↔test; revisión cruzada como gate; claims empíricos solo post-limpieza de cableado (C9); "lema" es pieza interna hasta el barrido de literatura; toda cifra citable sale del §9.

---

## 26. División del trabajo

**A (Vladimir):** A-M0 modelo · A-M1 $n{=}3$ · A-M2 $11/10/5$ · A-M3 A2 · A-M4 revisión formal A2 · A-M5 Lema A(i) · A-M6 (ii) · A-M7 (iii) · A-M8 coherencia · A-M9 corolarios+complejidad · A-M10 $S_0$ · A-M11 obstrucciones de S2 (a: tower; b: martingala) · A-M12 diseño de $\varphi_{\mathrm{virgin}}$ y $\varphi$ (estado enriquecido y acreditación parcial) · A-M13 acid test manual (G4a) · A-M14 interpretación del falsificador (incl. P21-A7 y flags de separabilidad) · A-M15 celda dinámico-binaria · A-M16 frontera · A-M17 formalizar la pregunta de garantía (§20: reducción o refutación; subir C5 a [DEMOSTRADO] o corregirlo; **interés prioritario de A**) · A-M18 validar literatura · A-M19 paper · A-M20 guion de sesión.

**B (Héctor):** B-M0 auditar APIs · B-M1 tests ancla + cobertura por intersección + G1 a ciegas · B-M2 consultas por demanda (perfilado) · B-M3 perfilado · B-M4 reproducibilidad · B-M5 interfaz de scorers ($S_0$; **no** S2 placeholder) · B-M6 rollout oracle + verificación Prop B · B-M7 harness acid test · B-M8 falsificador (con flags de separabilidad) · B-M9 candidatas $S_3$ (**si G4a aprobado**) · B-M10 rankings (G4b) · B-M11 adversaria dirigida (surrogate-mal / rollout-mejora / laminar-pierde / cruce-necesario / separación-con-greedy-competitivo) · B-M12 falsificador de garantía · B-M13 atlas extendido · B-M14 escala · B-M15 artefactos.

**Soporte IA:** C-M1 literatura · C-M2 árbitro de borradores · C-M3 harness de barridos · C-M4 lupa del paquete — subordinado a validación de A (fuentes, hipótesis) y B (código).

**Conjunto:** tras cada bloque, A revisa semántica de tests, B intenta falsificar enunciados; discrepancias bloquean cierre; no se avanza por calendario.

---

## 27. Dependencias

```text
Modelo normativo (G0)
├── Nota tensor (A2) ── Tests ancla (G1) ── Lema A (G2)
│                                            ├── Inferencia exacta
│                                            ├── Rollout oracle (G5)
│                                            └── Scorers S3 (G3)
├── Separación corregida ── Acid test (G4a→G4b) ── Celda dinámico-binaria
├── Biblioteca/jerarquía ── V*L / V*T ── Falsificador ── Garantía (§20)
└── Reproducibilidad ── Experimentos ── Atlas (G8) ── Paquete Francisco (G10)

S3 + Rollout + Acid test
└── Falsificación (G6) ── Selección ── Garantía (G7) ── Atlas extendido ── Escala (G9)
```

Ramas Reproducibilidad ∥ A2 ∥ Literatura ∥ Oráculo: paralelas (§1).

---

## 28. Orden temporal condicionado

**E1 Fundamentos** (modelo, A2, tests ancla, Lema A, literatura, reproducibilidad) → **E2 Objetivo** ($S_0$, muertes de S2, $\varphi$, rollout, G4a→candidatas→G4b) → **E3 Falsificación** (árbol, ranking, regret, adversaria, garantía) → **E4 Teoría** (frontera, dinámico-binaria, garantía) → **E5 Evidencia** (atlas extendido, escala, certificados si hacen falta). Sin fechas para teoremas; cierres por gate. La semana del 3-ago (§1) asigna esfuerzo sobre E1–E2; único compromiso fechado externo: la reformulación de la Conjetura C.

---

## 29. Riesgos y pivotes

**R1** S3 no recupera la separación → rollout truncado; identificar variable omitida; no extender atlas; es resultado en sí. **R2** S3 memoriza el ejemplo → adversaria, vecindad, familias múltiples. **R3** S3 cuesta como rollout → adoptar rollout y eliminar el surrogate. **R4** la garantía falla → contraejemplo + complementariedad + régimen o certificado (§30.1). **R5** laminaridad pierde mucho → cruces acotados; caracterización. **R6** la jerarquía domina la pérdida → separar selección de planificación; algoritmo de construcción. **R7** dinámico-binaria no cierra → cota + conjunta + limitación. **R8** la literatura absorbe la novedad → reposicionar como combinación/algoritmo/evidencia. **R9** reproducibilidad comercial → base libre + opcional licenciada. **R10** la sesión cambia prioridades → frentes modulares; la espina no se descarta sin decisión en §32.

---

## 30. Líneas subordinadas

**30.1 Certificados:** ruta de respaldo de la garantía (R4) y vara en escala ($V^{alg}\le V^*\le U_{pen}\le U_{PI}$); implementados y validados (106 instancias, cero violaciones); "greedy real ~0.98, certificado ~0.7 ⟹ el cuello es la demostración" se conserva. **30.2 Gibbs:** texto de C7; apéndice; no compite con A2/Lema A/surrogate. **30.3 MILP:** selector; no confundir MIP gap / error muestral / gap de política. **30.4 RL:** congelado; regresa solo con baseline exacto y evaluación justa. **30.5 Resolución/ruido:** extensión futura; el 84.5% del canal de tres niveles se conserva con alcance corregido. **30.6 Industria:** demo secundaria; nunca antes de cerrar el argumento.

---

## 31. Qué no se hará

Implementar S2 como placeholder; extender el atlas antes de G4b; entrenar RL; escalar sin candidato; afirmar submodularidad; llamar AS estándar a propiedades de $\Phi_b$ sin la reducción; afirmar cota 0.9; mezclar recompensas; citar Gibbs sin artefacto reintegrado; confundir laminar con matroide; optimizar tablas completas sin perfilado; más notebooks de resumen; claims sin trazabilidad; llamar "teorema" a corolarios de literatura o ingeniería de caché; ajustar $\lambda_b$ o $\varphi$ y reportar sobre el mismo atlas; usar cotas superiores (candidata D) como política sin calibración.

---

## 32. Registro de decisiones

Formato de entradas futuras: | Fecha | Decisión | Evidencia | Alternativas | Consecuencia | Revisar cuando |. Las filas originadas en sesión con Francisco marcan la Evidencia con origen "sesión AAAA-MM-DD" (§34-bis). La tabla semilla usa el formato compacto:

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-08-01 | Hard clearing; deducción informa, no acredita; **acreditación parcial** con $\kappa_{\mathrm{free}}/\kappa_{\mathcal T}$ como costo de acreditación completa | evita ambigüedad; el surrogate no valora acreditaciones inejecutables ni pierde el valor parcial |
| 2026-08-01 | Biblioteca ex ante para $V^{*,\mathcal L}$; ex post como extensión | coincide con el solver; Prop B válida |
| 2026-08-01 | Rollout como oráculo; **forma incremental** en scorer y oráculo (sin doble conteo $r$/$U(C)$) | bug detectado en revisión cruzada; alineación surrogate–rollout |
| 2026-08-01 | S2 descartado en ambas variantes; $S_1^{hard} = S_0$ | tower property / martingala; colapso definicional |
| 2026-08-01 | C5 como [DERIVACIÓN CONDICIONAL], $\le 0.434$ | modus tollens correcto, mapeo pendiente (A-M17) |
| 2026-08-01 | $\varphi$ con **estado local enriquecido** $\xi_D(H)$; $(D,c,b)$ es abreviatura | dos historias con el mismo triple difieren en valor acreditable |
| 2026-08-01 | **Knapsack declarado aproximación falsificable** (separabilidad del control no demostrada) | flags de separabilidad en el falsificador; sinergias multiátomo/átomo–virgen |
| 2026-08-01 | La garantía de $S_3$ **no presupone AS**: cinco preguntas separadas (§20) | $\Phi_b$ no es función adaptativa estándar sin reducción |
| 2026-08-01 | $\lambda_b = 1$ principal; calibración solo con holdout | anti-overfitting |
| 2026-08-01 | $\varphi_{\mathrm{virgin}}$ = política CBS restringida y declarada | cota inferior realizable, no Bellman escondido |
| 2026-08-01 | G4a/G4b; medianas como estadístico de caché; sin razones con denominador cero | rondas de verificación contra CSV |
| 2026-08-01 | IA como soporte subordinado; etapas por gate; supersede de 4 documentos | responsabilidad en A y B; anti-dispersión |
| 2026-08-02 | Capa de procedencia (§0-bis) con espina/entrada BC/formalización/extensiones; linaje $S_2\to S_3$ en §14; "muertes" → "obstrucciones [DERIVACIÓN → A-M11]"; vía submodular elevada (respuesta documentada en mínimo fuerte, A-M17 prioritario sin plazo, relajaciones convexas en §21); prioridad operativa en §1 | contraste del plan con la transcripción BC (rebote adversarial multi-LLM); separar lo pedido de lo propio sin atribución punto por punto; interés declarado de A en la vía submodular |

Entradas en formato completo (a partir del despacho de la sesión Lowell House; acta: `docs/notes/2026-08-02-sesion-francisco.md`):

| Fecha | Decisión | Evidencia | Alternativas | Consecuencia | Revisar cuando |
|---|---|---|---|---|---|
| 2026-08-02 | La tarea de sesión "evaluar V(T)=E_R[B(R)] en el ejemplo" se procesa como la variante S2-sobre-el-pool: evaluación analítica (extensión de A-M11) + diagnóstico etiquetado en el harness; NO se implementa como scorer (§31). Entregable: linaje sugerencia→obstrucción (tower: V=u·\|t\|·q en homogéneo — elige pool máximo, no extrae)→S3 como corrección realizable | sesión 2026-08-02 [38:56–43:07]; derivaciones A-M11 | implementarlo como candidato (viola §31); simular sin la obstrucción | compromiso fechado nuevo; §34 (2) anotada; preguntas (5)–(8) añadidas | próxima sesión |
| 2026-08-02 | Cuarentena del resultado "mejor desempeño en tasas altas" y auditoría previa a toda cita de C1 en prevalencia alta: trazas de decisiones en p>0.5 (¿primera prueba individual?) + contabilidad de empates en `showcase_regions.csv`, contra el teorema estático p>1/2 que Francisco coteja en el paper | sesión 2026-08-02 [12:23–16:26] | seguir citando C1 tal cual | tarea de auditoría en cadena B; posible refinamiento de lenguaje C1 | trazas + teorema confirmado |
| 2026-08-02 | La sesión valida de forma independiente el diseño S3: valor por componente y selección de rama (≈ φ(átomos)+knapsack, §14.5), costo log G (≈ caso base 14.9), bootstrap como crux cuya respuesta es φ_virgin (§14.7). Se añade a B-M11 la familia "separación-con-greedy-competitivo" | sesión 2026-08-02 [16:30–24:36, 32:15–35:39] | línea nueva (duplicaría §14) | sin milestone nuevo; refuerza A-M12/B-M9 | G4a |
| 2026-08-02 | El "fine-tune de parámetros para recuperar el planificador" [18:20–18:42] se ejecuta bajo el candado existente: diseño/validación/holdout adversarial congelado antes de evaluar; se reporta el protocolo de calibración junto al número. No se habilita ajustar-y-reportar sobre el mismo atlas (§31) | sesión 2026-08-02 | calibrar libre (viola §31); rechazar la sugerencia (innecesario) | la sugerencia se persigue con disciplina explícita; §34 pregunta (7) | G4a |
| 2026-08-03 | **Aprobación final del plan por A y B**: el estado pasa de "candidato" a "plan maestro vigente, aprobado por A y B el 3 de agosto de 2026" | aprobación comunicada por A | mantenerlo como candidato | el plan es norte vigente sin reservas; los supersedes de los 4 documentos quedan firmes | nueva revisión adversarial o supersede |
| 2026-08-11 | El encargo del 2026-08-02 se entrega en sesión: colapso de V(T)=E_R[B(R)] con cómputo exacto (Hechos 1-2, dos versiones c/u). La conversación valida la obstrucción, la refina (el score colapsado es exactamente **aditivo**: V(S)=Σuᵢqᵢ, ni siquiera submodular) y añade suboptimalidad en B=1 con q<0.5. Compromiso fechado 2026-08-02: **CUMPLIDO** | sesión 2026-08-11 [01:02–06:08]; derivaciones A-M11 | — | A-M11 presentado y validado en sesión; §14.4 confirmado con lectura de aditividad | — |
| 2026-08-11 | Directriz nueva: score de dos componentes — V(T) sin cambio + C(T) = E[# pruebas que greedy usa hasta extraer la utilidad viva], estimado por Monte Carlo sobre el posterior; regla de poda: descartar candidatos con C(T) > presupuesto restante. Se registra como **variante candidata de S₃** ("valor/costo con greedy como oráculo de costo"), no como reemplazo: misma moraleja de §14.4 (realizable bajo presupuesto). Responde preguntas (2) y (7) de §34 | sesión 2026-08-11 [07:03–08:07, 30:27–34:36] | implementarla como scorer sin pasar por diseño A-M12/G4a (viola §31); tratarla como sustituto de S₃ | alimenta A-M12 y B-M9; extensión menor de B-M6 (contador de pruebas-hasta-terminar en el rollout); preguntas (9)–(11) nuevas en §34 | G4a |
| 2026-08-11 | Prioridad de regímenes: primero u≡1 homogénea; después q homogénea con u heterogénea; el caso doblemente heterogéneo se pospone | sesión 2026-08-11 [20:57–21:42, 34:53–35:45] | atacar el caso general de entrada | ordena §23 y acota A-M12; no reordena gates | al cerrar G4b |

---

## 33. Entregables

**Documentos:** modelo normativo (§5, extraíble); nota del tensor; Lema A; registro de claims (§9, vivo); literatura; análisis de separación (con §18); análisis frontera; análisis de garantía (§20); paquete Francisco. **Código:** consultas por demanda; rollout oracle; interfaz de scorers; candidatas $S_3$ (con $\varphi_{\mathrm{virgin}}$ y estado enriquecido); falsificador (con flags de separabilidad); checker de garantía; atlas extendido. **Tests:** anclas; rollout; scorers; presupuesto; hard clearing; no doble conteo (incl. $r$/$U(C)$); acreditación parcial; falsificador; reproducibilidad. **Datos:** decisiones, valores, rankings, contraejemplos, tiempos, atlas. **Figuras:** tres pérdidas; mapa de régimen; surrogate vs. rollout; clases de acción; frontera; costo/calidad.

---

## 34. Paquete para Francisco

**Mensaje:** "La laminaridad parece conservar gran parte del valor del óptimo pequeño (razones 0.928 malla / 0.9069 adversaria) y permite inferencia exacta. La pérdida mayor aparece en la construcción de jerarquía y la miopía (greedy balanceado 0.747). Estamos usando rollout exacto para diseñar y falsificar un objetivo barato de utilidad realizable bajo presupuesto."

**Linaje de la sugerencia de la conversación:** la conversación planteó como candidato informal un potencial de "sanos esperados si después se testeara gratis" y la posible estructura submodular con relajaciones convexas. Ese candidato corresponde a $S_2$ y colapsa en ambas variantes (tower property / martingala, §14.4; pruebas completas en A-M11); su moraleja — el potencial debe ser realizable bajo presupuesto — es lo que produce $S_3$. La vía submodular directa tiene derivación condicional de imposibilidad bajo el mapeo natural (C5); quedan abiertas las versiones relajadas y la reducción (§20, preguntas 2–3). Se presenta como: sugerencia perseguida → obstrucción documentada → pregunta viva.

**Mostrar:** modelo; separación con ancla; tensor con hallazgo de costo; Lema A (estado exacto por partes); tres pérdidas; atlas; 0.9069; gap del greedy; acid test tal como salga; surrogate vs. rollout; resultado teórico o contraejemplo. **No como headline:** inventario del repo, bugs históricos, RL, notebooks, demo industrial, submodularidad como hecho.

**Preguntas:** (1) ¿aproximar $V^{*,\mathcal L}$ o $V^*$ a través de la clase? (2) ¿qué relajación expresa mejor "valor futuro": sanos identificables o utilidad acreditable con presupuesto residual? **[PARCIALMENTE RESPONDIDA — sesión 2026-08-02: preferencia revelada por sanos identificables vía posterior ($E_R[B]$); su argumento log-G apunta a acreditable-bajo-presupuesto; se cierra al presentar la obstrucción]** **[RESPONDIDA — sesión 2026-08-11: la dimensión del presupuesto es imprescindible ("algo que está faltando es incorporar el presupuesto restante"); forma acordada: valor + costo-en-pruebas]** (3) ¿priorizar la frontera $B{=}2/B{=}3$? (4) ¿qué garantía final valora más — aproximación, régimen, policy improvement, certificado — y la celda intermedia: ¿teorema o limitación?; en particular, ¿qué prioridad da a cerrar la vía submodular (la reducción de §20) frente a policy improvement o regret acotado? (5) criterio de éxito para "recuperar el planificador": ¿igualar el valor (≈0.806u) o replicar la política paso a paso? (6) ¿el paso miope maximiza V puro o $r + \lambda V$, y sobre qué familia de candidatos (biblioteca laminar o subconjuntos libres hasta $G$)? (7) el descuento por sanos detectados-no-localizados, ¿fijado por el costo $\log G$ (realizabilidad) o calibrado — y bajo qué protocolo? **[RESPONDIDA — sesión 2026-08-11: el costo se mide, no se fija ni calibra: $C(T)$ = E[# pruebas de greedy hasta finalizar], Monte Carlo sobre el posterior; esperanza de forma cerrada en homogéneo]** (8) en $B(R) = \sum_i u_i P(\text{sano}_i\mid R)$, ¿la suma excluye a los ya acreditados $C(H)$? (tal como se dictó los cuenta; choca con el requisito de no doble conteo de §14.1) (9) ¿existe forma cerrada de $C(T)$ con $u\equiv1$ (el tamaño de grupo greedy tiene expresión cerrada, función cóncava)? (10) ¿regla de decisión con dos puntajes: filtro de factibilidad $C\le b$, cociente $V/C$, o knapsack sobre candidatos? (11) ¿el costo como esperanza basta para la poda, o hacen falta cuantiles (colas de $C$ bajo presupuesto chico)?

---

## 34-bis. Ciclo de sesión con Francisco

**Protocolo (fijo).** Antes de cada sesión: el paquete y las preguntas de §34, congelados en el guion (A-M20). Después de cada sesión: acta en `docs/notes/AAAA-MM-DD-sesion-francisco.md` (transcript o resumen con citas fechadas; disciplina de atribución de la cabecera) y despacho en la tabla siguiente. Cada directriz se clasifica y se resuelve **solo** en estructuras existentes; esta sección nunca contiene objetivos propios.

**Tabla de despacho (buffer rotatorio — vive aquí solo hasta despacharse):**

| Directriz (cita/paráfrasis fechada) | Clasificación | Destino |
|---|---|---|
| — | confirma espina / ajusta prioridad / línea nueva / responde pregunta (N) de §34 | milestone (§26) · gate (§25) · fila §32 (origen "sesión") · entrada §0-bis |

**Compromisos fechados externos vigentes:** reformulación de la Conjetura C (P21-A8) · Francisco: cotejo del teorema estático $p>1/2$ en el paper. [Cumplido 2026-08-11: derivación + confirmación computacional de $V(T)=E_R[B(R)]$ en el ejemplo motivador — ver §32.]

**Reglas de choque:** ninguna directriz reordena gates ni redefine el modelo (§5) sin fila en §32; si contradice la espina, se procesa como R10; las preguntas de §34 respondidas se marcan y se reemplazan por las siguientes. Tras el despacho: la tabla se vacía (la historia queda en §32 y en git) y §1 se reescribe con la asignación de la semana entrante.

---

## 35. Criterios de éxito

**Mínimo fuerte (cumplimiento de la espina del programa, §0-bis):** modelo cerrado; A2; Lema A; rollout; $S_3$; acid test; falsificador; descomposición; reproducibilidad; **respuesta documentada a la vía submodular** (propiedad o reducción demostrada, limitación caracterizada o contraejemplo explicado). Los sobresalientes son contribuciones adicionales del equipo. **Sobresaliente A:** teorema de frontera ($B{=}2$) + caracterización de la ruptura en $B{=}3$. **Sobresaliente B:** garantía dentro de jerarquía o clase laminar. **Sobresaliente C:** propiedad de garantía formal (reducción de §20) + su cota. **Sobresaliente D (aunque la garantía formal falle):** contraejemplo mínimo explicativo + mecanismo de complementariedad + por qué un paso miope no basta + política planificada con mejora demostrable + certificado o régimen.

---

## 36. Conclusión rectora

El proyecto no intentará forzar una garantía de submodularidad. Intentará **entender y aproximar el valor de la planificación**.

> **Hipótesis principal:** una representación laminar exacta, combinada con un potencial de utilidad realizable bajo presupuesto — que valora el territorio virgen, los átomos condicionados y la acreditación parcial —, puede capturar una fracción sustancial del valor de rollout a menor costo.

La referencia será rollout. La prueba mínima será la separación. La evaluación separará planeación, jerarquía y laminaridad. La garantía se elegirá según lo que sobreviva: aproximación, régimen, policy improvement, certificado — o contraejemplo estructural.
