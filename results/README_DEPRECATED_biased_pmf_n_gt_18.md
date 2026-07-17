# DEPRECATED: columnas EU sesgadas en n > 18 (CSVs de 2026-03)

Los CSVs de esta carpeta fechados 2026-03 (`sprint3_*.csv`,
`overnight_*.csv`) fueron generados ANTES del fix de cableado de 2026-07
(rama `fix/inference-wiring`). En ellos:

- Toda columna `U_greedy_*` / `U_beta_greedy` con **n > 18** proviene de la
  recursion EU con pesos de rama Poisson-Binomial sobre marginales
  secuenciales: **sesgada** (+3-9% medido contra MC insesgado).
- Las filas con n = 19-20 estan etiquetadas implicitamente como "enum"
  aunque cruzan la frontera exacta (los gates viejos comparaban contra 20;
  la frontera real es `EXACT_PMF_MAX_N = 18`).
- Las filas con n <= 18 no estan afectadas.

Los runners corregidos emiten columnas `*_se` y una columna `estimator`
("exact" | "mc") para que ninguna fila futura sea ambigua. Para regenerar
las corridas largas con los estimadores insesgados, ver Task 13 del plan
`docs/plans/2026-07-16-inference-wiring-fixes.md` (corrida larga opcional).
