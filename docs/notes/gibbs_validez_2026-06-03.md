# Gibbs: hallazgo de correctitud y fix (stopgap) — 2026-06-03

_Corrige la afirmación de `estado_proyecto_2026-05.md` de que "Gibbs está validado
(64/64 tests)". La validación previa casi seguro ejercitó el **atajo exacto**, no
el muestreador MCMC real._

## El hallazgo

`augmented/bayesian.py :: gibbs_update` podía **contar perfiles de infección
inválidos** (inconsistentes con los conteos observados) en su estimado de
marginales, en el camino que **sí** usa MCMC (>7 agentes activos tras el
preprocesamiento; con ≤7 caía a conteo exacto).

Mecanismo (confirmado con código + 2 auditorías + segunda opinión de Codex):

1. **Init inválido.** El init greedy (asignaba infectados para cumplir cada
   conteo) **no garantiza** un perfil válido cuando los pools se solapan: solo
   añade, nunca quita sobre-asignaciones. → arrancaba inválido el **72%** de las
   veces (escenarios con pools grandes).
2. **Single-site congelado.** Desde un estado válido, el barrido single-site es
   un **no-op** para todo agente activo (su único valor consistente es el actual).
   La cadena solo se movía por swaps/bloques.
3. **Swaps no reparan de forma confiable.** Los bloques aceptan solo si todo
   queda válido; los swaps por test **solo** revisan los tests de los 2 agentes
   movidos (pueden dejar otras restricciones inválidas). Desde un init inválido
   casi nunca reparaban.
4. **Sin guard al recolectar.** La muestra se contaba **sin** checar validez.

### Evidencia (escenarios que sí usan el MCMC, >7 activos)

- Fracción de **muestras inválidas** contadas: media **~30%**, máx **100%**.
- **Mezcla** pésima: media 2–4 estados distintos visitados, **mín 1** (congelado).
- **Error marginal** vs exacto: media **~0.6**, máx **1.0**.
- Invariante duro `Σ marginales sobre un pool = r` (debe ser exacto porque todo
  perfil válido tiene `r` infectados ahí): el exacto lo cumple a 1e-15; el Gibbs
  lo violaba hasta **2.0–3.0** → prueba matemática de que contaba inválidos.

Scripts: `augmented/notebooks/gibbs_validity_audit.py` (función real) y
`gibbs_validity_audit2.py` (copia instrumentada: cuenta muestras inválidas y
mide mezcla). Test de regresión: `augmented/tests_gibbs_validity.py`.

## El fix (stopgap, no el rewrite)

En `gibbs_update`:

1. **Enumeración exacta sobre el conjunto ACTIVO** (`2^|activos|`, independiente
   de n) para `activos ≤ EXACT_ACTIVE_THRESHOLD = 16`. Correcto y barato; cubre
   todas las escalas reales (el DP topa en n≤14). De paso arregla un **bug
   latente**: el atajo anterior usaba `2^n`, así que colgaba con n grande aunque
   hubiera pocos activos.
2. **Init válido** por búsqueda de mínimos conflictos (`_find_valid_state`) para
   la rama MCMC.
3. **Guard de validez**: solo se cuentan muestras válidas; `total_samples` y la
   ventana de convergencia avanzan solo en válidas (refinamiento de Codex).
4. **Fallback exacto capado** (`≤ EXACT_ACTIVE_FALLBACK_CAP = 22`) si el MCMC no
   junta ninguna muestra válida — evita cuelgues (refinamiento de Codex).

### Verificación

- `tests_gibbs_validity.py`: **25/25** escenarios (>7 activos) — error 0.0,
  invariante 0.0 (antes: 25/25 fallaban, error 1.0, invariante 3.0).
- Auditoría #1 (función real): error 0.0/0.0 en 40 escenarios.
- Rama MCMC+guard (activos 17–18, n=22): invariante a **1e-16** (nunca cuenta
  inválidos); el error vs exacto puede seguir alto (~0.6) por mezcla limitada.
- Suite existente: **119/119** tests pasan, sin regresiones.

## Lo que SIGUE abierto

La **mezcla** del MCMC para `activos > 16` sigue siendo pobre: el guard garantiza
*consistencia* (nunca perfiles inválidos) pero no *exactitud* en ese régimen. Los
movimientos actuales son **no ergódicos**: un swap `1`↔`0` solo es válido si los
dos agentes tienen columnas de incidencia idénticas, así que solo permutan
etiquetas dentro de clases de agentes idénticos y **preservan el total de
infectados** — no pueden conectar perfiles válidos con distinto total (ej.:
tests `{0,1,2}=1`, `{2,3,4}=1` → `(0,0,1,0,0)` y `(1,0,0,1,0)` ambos válidos,
inalcanzables entre sí).

El arreglo de fondo (trabajo futuro, confirmado con Codex) es un muestreador con
**movimientos de base de Markov / Graver** para la fibra binaria del sistema
`A·z = r` (A = incidencia test–agente): vectores enteros `d` con `A·d = 0`
aplicados respetando factibilidad binaria. Los "ciclos alternantes" son solo un
subconjunto (preservan el total, insuficiente). Init por CSP/backtracking +
reseed cuando se visiten pocos estados distintos. En la práctica el proyecto no
llega ahí (activos ≤ 14).
