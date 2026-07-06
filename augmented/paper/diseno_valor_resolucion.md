# Diseño: El valor de la resolución en adaptive group counting bayesiano-adaptativo

**Fecha:** 2026-07-03
**Estado:** borrador de diseño, revisado adversarialmente (17 hallazgos aplicados) + addendum de Francisco (TV vs KL, mixing time). El marco está aprobado; pendiente de tu revisión antes de pasar al plan de implementación.
**Origen:** reunión con Francisco Marmolejo-Cossío (2026-07-02), tres direcciones propuestas.

---

## 1. Contexto y decisión

Francisco propuso tres direcciones:

- **D1 — Estructura:** modelar el problema como hipergrafo G=(V,E) con grado máximo K (número de pools en que participa cada sujeto); estudiar K=1, 2, 3 y ejemplos de rejilla (lattice 2×2, 3×3) con grupos entrelazados (p.ej. sujetos a,b,c,d probados como {a,b},{c,d},{a,c},{b,d}).
- **D2 — Información parcial:** tests que solo reportan un rango o bin del conteo de activos, no el conteo exacto ni el binario.
- **D3 — Benchmarks:** en counting estático, greedy es 2-aproximación de OPT; ver si el greedy dynamic-augmented está a un factor 2 del greedy estático.

Además, Francisco sugirió (addendum, §8): usar **distancia de variación total (TV)** en vez de KL para medir aproximaciones del posterior, y **explorar el mixing time** del generator de Gibbs.

**Decisión tomada:** paquete unificado, con **columna empírica** ("el valor de la resolución") como resultado central, y D1/D3 de apoyo. Calidad primero, sin deadline fijo.

**Corrección verificada contra la literatura (importante, afecta D3).** Lo que se enunció como "greedy estático = 2-aproximación de OPT" es un teorema real pero de otro problema y no aplica al nuestro:

- "greedy ≥ ½·OPT" es la cota de Fisher–Nemhauser–Wolsey (1978, parte II) para maximizar una función submodular monótona bajo una **matroide general**. Ese ½ también se cumple, como cota más débil, bajo cardinalidad; pero ahí la cota **ajustada** es más fuerte: **(1−1/e) ≈ 0.63** (Nemhauser–Wolsey–Fisher 1978, parte I). Es decir: ½ es la cota ajustada solo para matroides no uniformes; para presupuesto B (matroide uniforme / cardinalidad) la ajustada es (1−1/e), que además implica ≥½.
- Los pools no solapados (non-overlapping) sugieren una estructura de tipo partición, pero la disjunción entre pools elegidos no es en general un matroide de partición sobre el conjunto base (se parece más a una intersección de matroides / matching); no forzar esa analogía.
- Y lo decisivo: nada de esto aplica a nuestro problema, que es **adaptativo** (política contra política). Para greedy adaptativo vs OPT adaptativo se necesita **submodularidad adaptativa** (Golovin–Krause 2011), y esa propiedad **falla** en nuestro objetivo de limpieza (ver §7). Hoy no hay 2-aprox ni (1−1/e) para nuestro greedy; la única cota superior es la de información perfecta (hindsight, U_PI).

---

## 2. Pregunta, tesis y claim principal

**Pregunta.** Entre el test binario (1 bit: ¿hay algún activo en el pool?) y el conteo exacto (r = |t ∩ Z|), existe una escalera de resoluciones intermedias: cuantizadores por umbral que solo reportan en qué bin cae el conteo. **¿Cuántos niveles de resolución bastan para capturar (1−ε) del beneficio del conteo?**

**Tesis unificadora del paper.** Un mapa de las condiciones bajo las que el conteo aporta beneficio, gobernado por tres perillas: el **horizonte B**, el **grado estructural K** y la **resolución** del canal (niveles de bins). En los tres extremos el beneficio se apaga (B=1, K=1, 1 bin = binario); el paper caracteriza el interior. D2 es el eje de resolución (núcleo); D1 el eje K (apoyo); D3 explica por qué certificamos el greedy con hindsight y no con un factor de aproximación (subsección de método). Las distancias entre posteriores se miden con **TV**, no KL (§8), y el eje K tiene un segundo pago —la tratabilidad del posterior vía mixing time— además de la analizabilidad de la utilidad.

**Claim principal (a medir, no asumido).** *Hipótesis*: la utilidad como función de la profundidad de resolución tiene **rendimientos decrecientes** — los primeros niveles (distinguir 0 / 1 / ≥2) capturan la mayor parte del beneficio, y el conteo completo aporta poco margen. Es una hipótesis empírica a verificar en la escala runnable (§5), listada como pregunta abierta (§11). La novedad no es "la curva es cóncava" (folklore por el orden de Blackwell / rate–distortion), sino (a) su **forma cuantitativa concreta** en este modelo de limpieza dura con presupuesto, (b) la restricción estructural "aislar {0}" que la hace válida, y (c) el mapa de encendido/apagado del beneficio a lo largo de B, K y resolución.

**Colapso en B=1 (corolario, no solo dato).** Con B=1 el resultado del único test no condiciona ninguna decisión posterior, así que el óptimo = max_t E[utilidad de limpieza de t] es idéntico en los tres canales (el valor de la información es nulo sin una decisión que informar). El chequeo numérico lo confirma, pero el origen del resultado es el modelo, no la medición.

---

## 3. Modelo formal

Sea un pool t con |t| miembros y perfil de estado latente Z. El conteo es r = |t ∩ Z| ∈ {0, 1, …, |t|}.

Un **cuantizador por umbral** Q parte {0, 1, …, |t|} en bins **contiguos**; el test devuelve el bin que contiene a r. La familia central del paper es la de **truncamiento** min(r, k), que reporta 0, 1, …, k−1 exactos y agrupa "≥ k":

- k = 1: Q = {0} / {≥1}  →  régimen **binario**.
- k = |t|: cada valor su propio bin  →  régimen de **conteo** (augmented).
- 1 < k < |t|: resoluciones intermedias (*semi-quantitative group counting*, SQGT).

**Restricción clave:** Q debe **aislar el {0}** (el bin del conteo 0 es exactamente {0}, con su propia etiqueta), para preservar la limpieza dura del modelo: la utilidad de un sujeto solo se acredita al colocarlo en un pool cuyo resultado observado sea exactamente r = 0. La cadena de truncamiento min(r,k) aísla {0} por construcción para todo k ≥ 1.

---

## 4. Teorema ancla (D2): monotonía por refinamiento

> **Lema.** Sea Q′ un refinamiento de Q (cada bin de Q es unión de bins de Q′). Entonces U_Q ≤ U_Q′ como valor de la política óptima. En particular, a lo largo de la cadena anidada de truncamiento min(r,k), la utilidad es **no decreciente en k**: U_binario = U_{k=1} ≤ U_{k=2} ≤ … ≤ U_conteo.

**Consulta (garbling determinista, etapa a etapa).** La recompensa total es Σ_i u_i·1[i cae en un pool jugado con r = 0]; es función únicamente de (pools jugados, Z) e **invariante al canal**, precisamente porque todo cuantizador en juego aísla {0} y la acreditación solo mira "r = 0". La política óptima bajo Q se reproduce bajo Q′: en cada etapa la política fina juega los **mismos** pools y aplica el coarsening determinista Q′→Q (unión de bins) a lo que observa, descartando la información extra; induce así exactamente la Q-historia y la misma recompensa en toda realización de Z. Luego el valor bajo Q′ no puede ser menor. El argumento es válido en el setting secuencial porque el refinamiento es un **garbling determinista** (coarsening de particiones), no una dominancia de Blackwell estocástica general (cuya extensión secuencial no es inmediata).

**Qué desigualdad usa qué hipótesis.** U_Q ≤ U_conteo vale para **todo** cuantizador Q (la identidad refina cualquier partición). Solo U_binario ≤ U_Q usa "aislar {0}" (equivale a que Q refine al binario).

**Caveat (necesario).** Si Q **funde {0,1}** en un mismo bin, deja de certificar la limpieza: bajo limpieza dura estricta ("acreditar exige observar el bin exactamente = {0}"), U_Q **colapsa a 0 sistemáticamente** (no "puede fallar"), y U_binario ≤ U_Q falla. En términos informacionales, para |t| ≥ 2 el binario y ese Q son además Blackwell-incomparables. Implicación de implementación: el solver debe **asertar** que los umbrales aíslan {0} (bin del conteo 0 = su propia etiqueta), para blindar el `if r == 0`, que si no dejaría de limpiar en silencio. `reconstruct()` debe reflejar el mismo binning si se necesita la política, no solo el valor.

**Atribución.** El lema es una instancia del teorema de comparación de experimentos de Blackwell (1951): un garbling no puede aumentar el valor. Lo propio no es la monotonía en sí, sino su aplicación al modelo de limpieza dura con la restricción "aislar {0}".

---

## 5. Experimento central: la curva de resolución

**Código nuevo (no es reuso; es la mayor parte del esfuerzo).**

- `bin_of(r, umbrales)`: función **escalar** que mapea un conteo r a su bin. Es lo que necesita el DP, que ramifica perfil-a-perfil (`neg_list`/`pos_list`) y nunca forma una PMF. Va en [solver.py](../solver.py).
- Parámetro `umbrales` en `solve_optimal_dapts` (y en su reconstrucción de política), que ramifica sobre bins en vez de sobre el conteo completo.
- `quantize(pmf, umbrales)`: agrupa una PMF del conteo por bins. Pertenece al **otro** camino de código (greedy / utilidad esperada), no al DP. Nota: `exact_pool_pmf` vive en [bayesian.py](../bayesian.py) (~L214), no en solver.py, y el DP no la invoca.

**Escala (todo corre en la M4).**

- **Régimen exacto:** DP hasta el límite práctico (~N = G = 5, B = 3; N ≤ 6 en regímenes ligeros). Ahí U_binario, U_Q y U_conteo son exactos y la curva es exacta. Es el régimen del resultado comprometido.
- **Régimen grande (solo cualitativo):** la vía "greedy + hindsight a mayor n" es **código nuevo**, no reuso: el greedy exacto por esperanza es O(2ⁿ) y solo escala vía Monte-Carlo por mundo; hay que construir el harness MC, el plumbing del cuantizador dentro del greedy `simulate`, y U_PI muestreado. La maquinaria del posterior a escala (Gibbs, mixing) se discute en §8. Opción honesta: acotar la curva empírica al régimen exacto (n ≤ 6) y presentar el tramo grande solo como esquema cualitativo, no como medición.

**Cota superior U_PI (hindsight): también código nuevo (~20–30 líneas).** No existe en el repo (`hindsight`/`U_PI`/`clairvoyant`: 0 coincidencias en .py); se ha calculado antes de forma ad hoc en notebooks. Definición: U_PI = E_Z[ suma de las top-(B·G) utilidades entre los limpios en Z ], enumerando 2ⁿ en el régimen exacto y por Monte-Carlo a escala grande. Añadir `solve_hindsight(p, u, B, G)` al inventario (§10).

**Costo del barrido.** Ramificar por bins **no** abarata el solve: medido (n=6, G=5, B=3), cap=1 (binario) ≈ 0.74 s, pero cap = 2..5 todos ≈ 1.3 s (en cuanto hay un bin "≥2" el espacio de estados memoizado es tan fino como el del conteo). La curva cuesta ≈ (#cuantizadores) × (DP-conteo): un barrido de horas a n=6, no "mínimo". Podar puntos redundantes: los caps altos colapsan (cap ≥ 3 idénticos en el prototipo).

**Barrido.** Cadena de truncamiento min(r,k): k=1 (binario) → k=2 → k=3 → … → conteo total; sobre varios regímenes de prevalencia; B ∈ {1, 2, 3}.

**Métricas.** U_{k} vs profundidad k (la curva); fracción capturada (U_k − U_binario)/(U_conteo − U_binario); verificación numérica de la monotonía U_binario ≤ U_k ≤ U_conteo (control del lema §4); colapso en B=1 (control del corolario §2).

**Entregable.** La curva de resolución, evidencia de sus rendimientos decrecientes, y el acople con el horizonte.

---

## 6. Perilla K (D1, apoyo): colapso K=1 y resultado K=2 (array)

K = grado de un sujeto = número de pools en que participa.

**Pendiente de definir con Francisco (bloquea el enunciado, no el experimento):** si K acota (a) la familia de pools disponibles, (b) el grado en el árbol adaptativo, o (c) el grado en el camino jugado. Las conclusiones difieren según cuál. Arrancamos con (a), la más limpia.

**Código nuevo (no es reuso del enmascarado VW).** [solver.py](../solver.py) y [classical_solver.py](../classical_solver.py) fijan `pools = all_pools(...)` hardcodeado y no aceptan un parámetro `pools`; `vw_restrict.py` es un ranking heurístico que ni siquiera llama a `solve_optimal_dapts`; `all_pools_from_mask` restringe por vértices activos, no por grado de hiperarista. Hay que: añadir un parámetro `pools=None` a `solve_optimal_dapts` (y replicarlo en la reconstrucción de política) y definir el generador de pools del hipergrafo de grado ≤ K (para los arrays 2×2/3×3 basta enumerarlos a mano).

**Resultados.**
- **Colapso K=1** (grupos disjuntos): el conteo no separa y greedy = OPT. Segundo punto donde el beneficio se apaga. Verificado con el solver restringido.
- **K=2 (array/grid):** el ejemplo entrelazado 2×2 ({a,b},{c,d},{a,c},{b,d}) y su extensión 3×3 (filas + columnas). Resultado empírico: dónde y cuánto reaparece el beneficio del conteo bajo grado 2.

**Caveats honestos.**
- Acotar K solo **baja** el óptimo alcanzable (U*_K ≤ U*); lo que compra es **analizabilidad** (concentración, certificados) y, vía mixing, **tratabilidad del posterior** (§8), no desempeño.
- El colapso K=1 es dependiente del modelo de limpieza dura: bajo un objetivo de clasificación, el conteo sí separa incluso en un pool aislado con prior asimétrico.
- Una **cota teórica** para K=2 es un objetivo abierto/difícil. Va como *stretch goal*, no como bloqueante: el resultado comprometido es el **empírico** del array.

---

## 7. Benchmarks honestos (D3, subsección de método)

Sostiene la decisión metodológica de certificar el greedy con hindsight en vez de con un factor de aproximación.

**Lo que sí se sabe.** El predecesor estático (*Welfare-Maximizing Adaptive Group Counting*, arXiv:2206.10660) establece garantías del greedy en el régimen **estático** (cotas overlapping vs non-overlapping, heurística near-óptima). Para maximización submodular monótona por cardinalidad, la cota clásica es (1−1/e) (Nemhauser–Wolsey–Fisher 1978), no ½. El hallazgo de que **el greedy simple captura gran parte del beneficio** es del paper base dinámico (Lopez et al. 2026); lo citamos como tal, no como resultado propio.

**Lo que no.** No existe 2-aprox ni (1−1/e) para el greedy **adaptativo** de este repositorio. La única cota superior disponible es hindsight (información perfecta) U_PI, válida a cualquier escala (U_PI ≥ U_DA para todo n).

**Resultado conteo-cero (a reproducir con nuestro código).** El objetivo de limpieza **no es adaptive-submodular**. Contraejemplo con prior independiente, válido en binario y en conteo: individuos {1,2}, p = (0.99, 0.5), u = (1, 100), acción e = {1,2}:

    Δ(e | ∅) = P(ambos limpios)·(u₁+u₂) = 0.01·0.5·101 = 0.505
    Δ(e | {1}→conteo-cero) = P(2 limpio)·u₂ = 0.5·100 = 50

Como 0.505 < 50, la ganancia marginal **crece** al condicionar en más observaciones, violando los rendimientos decrecientes que exige la submodularidad adaptativa. Esto **da el mecanismo detrás** del patrón empírico "el hueco greedy-vs-OPT crece con B" (el contraejemplo es de una ronda; el crecimiento monótono con B queda como observación empírica a reproducir, §11).

**Nota de encuadre.** No prometer el teorema conteo-no-cero adaptativo (muerto por el contraejemplo) ni vender el lema submodular estático como contribución propia (es NWF 1978, y su versión aplicada es del predecesor estático). El aporte teórico ambicioso, si se persigue, es un **surrogate** adaptive-submodular (estilo EC2) o una cota vía submodularity-ratio/curvatura — terreno de Francisco.

---

## 8. Posterior a escala: TV como métrica y mixing time del Gibbs (addendum de Francisco)

Sección de apoyo/método. Toca la maquinaria que sostiene el régimen grande (§5) y da un segundo pago al eje K (§6).

**TV, no KL, para medir la aproximación de independencia.** El greedy puntúa con el **producto de marginales** (aproximación de independencia) en vez del posterior conjunto exacto; el "hueco de independencia" mide qué tan lejos queda esa aproximación. Ya lo medimos con **distancia de variación total (TV)**: `independence_gap.py:tv_distance` compara el PMF conjunto exacto (`exact_pool_pmf`, enumerando 2ⁿ mundos) contra el de independencia (Poisson-Binomial sobre marginales). KL no se usa en ninguna parte del repo. La observación de Francisco **fundamenta** esta elección: para una función de valor **acotada** (la utilidad está acotada por Σuᵢ), TV **acota directamente** el error de utilidad,

    |E_P[v] − E_Q[v]| ≤ (max v − min v) · TV(P, Q),

mientras que KL es no acotada y asimétrica, y solo se relaciona con TV vía Pinsker (TV ≤ √(KL/2)). Consecuencia concreta: usar TV de forma **consistente** — no solo para el hueco de independencia sino como métrica de convergencia del Gibbs. Hoy `gibbs_analysis.py` reporta el error **máximo de marginales** vs el exacto; conviene un diagnóstico basado en TV (del PMF de resultado de pool y/o de las marginales).

**Mixing time del Gibbs (thread teórico para Francisco).** El posterior a escala grande se aproxima con un generator de Gibbs (`bayesian.py:gibbs_update`): descompone en componentes conexas, resuelve exacto las de ≤ 16 activos y usa MCMC (movimientos de base de Markov por caminos alternantes, aceptación Metropolis por razón de priors) en las demás. Hoy **no hay cota ni diagnóstico formal de mixing time** — solo error empírico vs iteraciones (`gibbs_analysis.py`). Explorar el mixing time es lo que permitiría pasar el régimen grande de "cualitativo" a **certificado**: cuántas iteraciones bastan para que el posterior aproximado quede a TV ≤ δ del exacto. **Conexión con D1:** un **grado acotado K** suele implicar mezcla más rápida (condiciones tipo Dobrushin / gap espectral en grafos de interacción de grado bajo), de modo que la perilla estructural K compra no solo analizabilidad de la utilidad (§6) sino también **tratabilidad del posterior**.

**Encuadre honesto.** Usar TV de forma consistente es un arreglo de método barato y correcto. El mixing time es investigación abierta y de mayor riesgo: un *stretch* que, si sale (idealmente ligado a K), le daría al paquete un segundo resultado con dientes. No condicionamos el núcleo (la curva de resolución) a que salga.

---

## 9. Posicionamiento

**Vecino directo (punto de partida).** Lopez, Marmolejo-Cossío, Tello Ayala, Parkes, *Dynamic Welfare-Maximizing Adaptive Group Counting* (arXiv:2601.22419, 2026) — Francisco es coautor. Fija el modelo dinámico, el welfare de limpieza y el hallazgo de que el greedy simple captura gran parte del beneficio en presupuestos bajos. **Nuestro delta:** ellos operan en el **canal binario**; nosotros añadimos la **escalera de resolución** binario → cuantizador → conteo y el **mapa de cuándo el conteo ayuda** (perillas B, K, resolución). El predecesor estático es *Welfare-Maximizing Adaptive Group Counting* (arXiv:2206.10660).

**Competencia reciente (jun-2026): threshold group counting.** van der Hofstad–Müller–Riddlesden; Coja-Oghlan et al. Un test de umbral (¿r ≥ u?) es precisamente un **cuantizador grueso del conteo**, es decir un caso de nuestra propia escalera; el solape no es de canal. El deslinde correcto es de **objetivo y régimen**: ellos son no-adaptativo, recuperación exacta, min-tests, columna constante; nosotros somos **adaptativo, welfare, presupuesto B, prior heterogéneo, limpieza (dura) parcial**.

**Encuadre del canal.** El conteo r = |t ∩ Z| es el canal aditivo de *quantitative group counting* / coin-weighing (Erdős–Rényi 1963; Lindström 1965; survey Aldridge–Johnson–Scarlett 2019); los bins intermedios son *semi-quantitative group counting* (Emad–Milenkovic) y *threshold GT* (Damaschke). Estas fijan que el canal no es invención nuestra; la contribución es el diseño secuencial bayesiano de welfare bajo ese canal, y la caracterización de la escalera de resolución.

---

## 10. Entregables y secuencia sugerida

1. **Código nuevo.** `bin_of(r, umbrales)` (escalar, para el DP) + parámetro `umbrales` en `solve_optimal_dapts` y su reconstrucción (D2). `quantize(pmf, umbrales)` para el camino greedy. `solve_hindsight(p, u, B, G)` (cota U_PI). Parámetro `pools=None` en el solver + generador de pools de grado ≤ K (D1). Diagnóstico de convergencia del Gibbs basado en TV (upgrade de `gibbs_analysis.py`) (§8). Harness Monte-Carlo solo si se persigue el tramo grande.
2. **Experimentos.** Curva de resolución en régimen exacto (D2, núcleo); colapsos B=1 y K=1; array K=2 (D1); reproducción del contraejemplo D3.
3. **Teoremas.** Monotonía por refinamiento / garbling (§4). Enunciado y contraejemplo de no-adaptive-submodularity (§7).
4. **Thread teórico (stretch).** Exploración del mixing time del Gibbs, idealmente ligado a K (§8).
5. **Notebook de figuras** en el estilo de los existentes (figuras en celdas, sin build).
6. **Prosa del paper** (voz sobria, estilo tesis — usar la skill de estilo de redacción al escribirla).

Orden: `bin_of` + parámetro `umbrales` en el DP → curva de resolución (núcleo D2) → controles (colapsos B=1/K=1) → solver-K + array K=2 → `solve_hindsight` + sección honesta D3 → diagnóstico TV + (stretch) mixing → posicionamiento y figuras.

---

## 11. Riesgos y preguntas abiertas

- **Definición de K** (§6): decidir con Francisco antes de enunciar teoremas de D1.
- **¿La curva tiene rendimientos decrecientes?** Hipótesis, no probada; medirla en el régimen exacto antes de enunciarla (§2).
- **El hueco greedy-vs-OPT crece con B:** empírico, a reproducir; el contraejemplo (§7) da el mecanismo, no la monotonía en B.
- **Cota teórica K=2:** abierta; queda como stretch, no como compromiso.
- **Mixing time del Gibbs** (§8): sin cota formal hoy; explorar si el grado acotado K da una cota de mezcla. Stretch teórico con Francisco.
- **Reproducir localmente** los contraejemplos numéricos (clearing no-AS; hindsight; colapso K=1) con la maquinaria del repo antes de citarlos.
- **Leer en detalle** los papers de jun-2026 (solo se vieron abstracts) para deslindar con nitidez.
- **Surrogate adaptive-submodular** (§7): pieza que decide si D3 tiene teorema conteo-no-cero. Pregunta directa para Francisco.

---

## 12. Qué NO haremos (YAGNI)

- No perseguir el teorema conteo-no-cero del greedy adaptativo (refutado).
- No presentar el lema submodular estático (NWF 1978) como contribución propia, ni revender el hallazgo del greedy del paper base como nuestro.
- No condicionar el paper a una cota teórica de K=2 ni a que salga la cota de mixing time.
- No empujar la escala más allá de lo que corre en la M4 para los resultados exactos.

---

## Referencias (a confirmar formato al escribir el paper)

- N. Lopez, F. Marmolejo-Cossío, J. R. Tello Ayala, D. C. Parkes. *Dynamic Welfare-Maximizing Adaptive Group Counting.* arXiv:2601.22419 (2026). [paper base dinámico]
- *Welfare-Maximizing Adaptive Group Counting.* arXiv:2206.10660 (2022). [predecesor estático]
- G. Nemhauser, L. Wolsey, M. Fisher. *An analysis of approximations for maximizing submodular set functions — I* (1978, cardinalidad, 1−1/e); Fisher–Nemhauser–Wolsey, parte II (matroide general, ½).
- D. Golovin, A. Krause. *Adaptive Submodularity* (2011).
- D. Blackwell. *Comparison of Experiments* (1951).
- Desigualdad de Pinsker (Csiszár–Kullback–Pinsker) — relación TV ↔ KL.
- D. Levin, Y. Peres, E. Wilmer. *Markov Chains and Mixing Times* (2ª ed., 2017); condición de unicidad de Dobrushin para mezcla rápida en interacción de grado bajo.
- P. Diaconis, B. Sturmfels. *Algebraic algorithms for drawing from conditional distributions* (1998). [bases de Markov, usadas por el generator de Gibbs]
- Erdős–Rényi (1963); Lindström (1965); M. Aldridge, O. Johnson, J. Scarlett, *Group Counting: An Information Theory Perspective* (2019). [canal QGT]
- A. Emad, O. Milenkovic, *semi-quantitative group counting*; P. Damaschke, *threshold group counting*.
- van der Hofstad–Müller–Riddlesden; Coja-Oghlan et al. (2026). [competencia threshold GT, por leer en detalle]
