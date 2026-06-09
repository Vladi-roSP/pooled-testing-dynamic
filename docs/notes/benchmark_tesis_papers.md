# Benchmark: tesis y papers (para validar la reimplementación)

_2026-06-04. Extracción de complejidad, garantías teóricas, limitaciones y números
clave de las tres fuentes, como referencia contra la cual validar el proyecto
augmented. Generado leyendo los PDF; las citas remiten a sus secciones._

## Las tres fuentes

1. **Finster, González Amador, Lock, Marmolejo-Cossío, Micha, Procaccia —
   "Welfare-Maximizing Pooled Testing"** (arXiv:2206.10660, EC'23). El paper
   **fundacional y estático** (un solo round). Aquí viven las garantías teóricas.
2. **Lopez, Marmolejo-Cossío, Tello Ayala, Parkes — "Dynamic Welfare-Maximizing
   Pooled Testing"** (paper). La **extensión dinámica** del anterior.
3. **Tesis de Nicholas R. Lopez** (Harvard, mayo 2025). La versión larga del
   paper dinámico, con todos los experimentos y pseudocódigo (Apéndice A).

## Convención del modelo (ojo al comparar)

En Finster/tesis: `u_i` = utilidad por quedar en un test **negativo**; `p_i` =
probabilidad de estar **SANO**. Bienestar de un plan T: `u(T) = Σ u_i · P_i^T`,
con `P_i^T` = prob. de quedar en ≥1 test negativo. Un pool es negativo sii todos
sanos. El **proyecto augmented invierte la convención** (`p_i` = prob. infectado)
y además usa el **conteo exacto** `r=|t∩Z|` en vez de positivo/negativo binario;
esto cambia la inferencia (ver más abajo).

## Complejidad (Big-O) — benchmark de "complejidad de tiempo"

| algoritmo | tiempo | fuente |
|---|---|---|
| Bayes exacto (posterior) | `O(B·G·2^(BG))` | tesis §2.2.1 |
| Gibbs (marginal) | `O(m·N·(G+w)) → O(N·G)` con m,w ctes | tesis §2.2.2 |
| Non-pooled estático | `O(N log B)` | tesis A.3 |
| Conic single-test (FPTAS) | `O(N^5)`, ≥1−1e−7 del óptimo | Finster; Goldberg-Rudolf |
| Greedy non-overlapping | `O(B·N^5)` | tesis A.4 |
| Overlapping óptimo (estático) | `O(C(S,B)·B·G·2^(B(G+1)))`, `S=Σ_{k≤G}C(N,k)` | tesis A.6 |
| MILP non-overlapping | worst-case `O(2^poly(N))`, factible con Gurobi | tesis §3.0.5 |
| **DP óptimo dinámico** | `O((2S)^B·T)=O(S^B·B·G·2^(B(G+1)))≈O(N^(BG)·…)` | tesis §4.3 |
| **Greedy dinámico** | `O(2^B(N^5+N·G)) ⊂ O(2^B·N^5)`; `O(B·N^5)` si solo se sigue el resultado real | tesis §4.4 |
| Supervised (FFN) | salida `(2^B−1)·N`; labels via DP óptimo (inviable a escala) | tesis §5.1 |
| PPO | acción cruda `2^((2^B−1)·N)` (doblemente exp.); reducida con buckets | tesis §5.2 |

## Garantías teóricas — benchmark de "demostraciones matemáticas"

Las garantías **vienen de Finster et al.** (paper 1), no de la tesis (que es
aplicada y no enuncia teoremas formales propios):

- **Cota overlapping vs non-overlapping:** un plan non-overlapping óptimo rinde
  ≥ `1/4` del overlapping óptimo. Cotas finas del ratio de overlap: `R(2)=7/6`
  (probado), `R(3)≤7/3`, `R(4)≤15/4`, `R(B)≤4` para todo B; cota inferior `7/6`.
  El gap (4 vs 7/6) sigue **abierto** (conjeturan ~7/6).
- **Greedy estático:** `(1−ε)/5` del óptimo non-overlapping (≥20%); se cree
  holgada. **OrderedGreedy** (utilidades homogéneas): `1/e` (~37%), **la única
  cota probada ajustada**.
- **Conic single-test:** ≥ `1−1e−7` del óptimo single-test.
- **MILP:** rinde `1−Δ·B` del óptimo, `Δ ∈ O(e^R((S−R)/K)^2)`.
- **Óptimo dinámico ≥ óptimo estático** (dominancia débil): todo plan estático es
  un caso particular de plan dinámico (tesis §4, por Ley de Esperanza Total).
- **NO hay** submodularidad ni garantía `(1−1/e)` en ninguno de los tres
  documentos, y **no se prueba NP-hardness** (solo se menciona el CSP de
  Aprahamian et al.). Esto deja **abierto** el hilo VW / super-nodos de
  Marmolejo: darle al greedy dinámico una garantía tipo `(1−1/e)`. Ver
  `vw_submodularity.md`.

## Limitaciones declaradas — benchmark de "limitaciones"

- **Greedy dinámico usa solo marginales, no la conjunta** → puede calcular mal la
  prob. de un test negativo y elegir pools subóptimos (tesis Ejemplos 4.3, 4.4;
  en 4.4 el pool óptimo daría +16.6% en el 2º test). Es la limitación conceptual
  central, y coincide con lo que el proyecto reimplementó con conteos.
- **Inferencia Bayesiana exacta conjunta es intratable** (dependencias por
  solapamiento crecen exponencialmente); por eso Gibbs aproxima marginales.
- **Tradeoff de varianza:** dinámico/overlapping suben la varianza del bienestar
  con B y G; el percentil 20 del greedy cae por debajo del MILP para B∈{3,4,5},
  G=5; welfare cero en 0.0278% vs 0.0124% del MILP.
- **Independencia** de estados de infección (justificada por lockdown); no modela
  correlaciones.
- **Óptimo dinámico inviable** salvo instancias muy chicas (ver límites abajo).
- **ML no competitivo aún** (Supervised es proof-of-concept; PPO ~29,479 h-CPU y
  aún por debajo del greedy).
- **Secuencialidad:** el dinámico exige esperar resultados antes del siguiente
  pool (ventana de validez de muestras: VIH ~24h, Hep. C hasta 168h).

## Límites de tamaño y números a reproducir — benchmark de validación

- **Óptimo dinámico factible:** N≤10, G≤10, B≤5; data a gran escala solo
  **N=G=5, B=3**. (No "n=15".) N=G=10,B=5 tomaría "años" aun en clúster.
- Instancias chicas: N=G=3,B=2 y N=G=5,B=3 (1,000 c/u). Grandes: N=50,B=5,G∈{3,5}
  (10,000 c/u), `u_i∈{1,2,3}`, `p_i~U(0,1)`.
- **Greedy dinámico vs MILP** (N=50): **+2.76%** bienestar (G=5), **+1.24%** (G=3);
  gana al MILP en **53.87%** de casos.
- Greedy dinámico ≈ **99.0%** del overlapping (N=G=3,B=2); **>99.5%** (N=G=5,B=3).
- Ejemplo (dinámico vs estático óptimo): **+15.4%** (0.2846 vs 0.2466).
- Ejemplo 2 (gap del greedy): greedy **1.5** vs óptimo dinámico **1.75**.
- Tabla 6.1 (bienestar promedio, N=G=5,B=3): Non-Pooled 1.10, Greedy Non-Ov 1.03,
  Non-Ov 1.13, Overlapping 1.14, Greedy Dynamic 1.13, SL Dynamic 0.81, Optimal
  Dynamic 1.15.
- Tiempos Finster (AWS c6g.8xlarge, Gurobi): MILP de 292 ms (B=2) a ~4.7 días
  (B=34); GREEDY de ~49 ms a ~12.6 min en el mismo rango.

## Notas de divergencia del proyecto augmented (a tener en cuenta)

- El proyecto usa **conteo exacto** (r), no binario → modelo más informativo que
  el de los papers; las complejidades del posterior cambian.
- `p_i` = infectado (no sano); el `bayesian_update_by_counting` del repo enumera
  `2^n` sobre toda la población, menos eficiente que el `2^(BG)` de la tesis.
- El bug de Gibbs que se halló y arregló (contar perfiles inválidos) es propio de
  la variante augmented con conteos; los papers solo usan Gibbs para marginales
  binarias. Ver `gibbs_validez_2026-06-03.md`.
