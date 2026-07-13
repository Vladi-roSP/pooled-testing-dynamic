"""Build notebook_separacion.ipynb — presentación didáctica de la separación
(la jerarquía U_single <= ... <= U_max y el beneficio del conteo) para discutir
con Francisco. Construido sobre el código CORREGIDO (hierarchy_experiment.py) y
los resultados en results/hierarchy/.

Incorpora una revisión pedagógica (Codex, 2026-06): ejemplo que SÍ separa
augmented de binario, mini-argumento de por qué la cadena vale por construcción,
honestidad sobre empates y distribución (no solo medias), Umax como cota y no como
"otra tecnología", y salvedades de identificación en la curva de crecimiento.

Run:
    python augmented/notebooks/build_separacion_notebook.py
Then execute in-place:
    jupyter nbconvert --to notebook --execute --inplace \
        augmented/notebooks/notebook_separacion.ipynb
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "notebook_separacion.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}


def md(src):
    nb.cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    nb.cells.append(nbf.v4.new_code_cell(src))


# ===================================================================
md(r"""# Notebook 14 - La separación, paso a paso

**Objetivo.** Presentar con cuidado la jerarquía
$U^{\text{single}} \le U^s_{NO} \le U^s_{O} \le U^D \le U^D_A \le U^{\max}$
y, sobre todo, el eslabón nuevo $U^D \le U^D_A$: el valor de contar.

**Cómo leerlo.** No empezamos por la tabla. Subimos en capas: el mecanismo en una
instancia, por qué la cadena vale siempre, la jerarquía vista en instancias
concretas, el promedio como cascada, y el crecimiento del beneficio con la escala,
mostrando su distribución y no solo la media. La tabla agregada va al cierre.

**Procedencia.** Todo se regenera aquí desde `hierarchy_experiment.py` (código
corregido) y los CSV por instancia en `results/hierarchy/`. No hay números
escritos a mano.
""")

# ----- imports / setup -----
code(r"""import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(''))))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import augmented
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(augmented.__file__)))

from augmented.core import mask_from_indices
from augmented.bayesian import bayesian_update_by_counting
from augmented.hierarchy_experiment import hierarchy_for_instance, CHAIN

print('repo root:', ROOT)
print('cadena:', ' <= '.join(CHAIN))""")

# ===================================================================
md(r"""## Capa 1a — El mecanismo: una deducción que el binario NO puede hacer

El ejemplo tiene que separar de verdad el conteo del binario; si no, no consulta
nada. Tres personas, dos tests: $\{0,1,2\}=1$ y $\{1,2\}=1$. El conteo dice que
hay exactamente un activo en todo el grupo y exactamente uno entre $\{1,2\}$;
ese único activo vive en $\{1,2\}$, así que **0 queda limpio con certeza**.

Un planificador binario ve los mismos dos pools como conteo-no-cero y conteo-no-cero, y de ahí
no puede concluir nada sobre 0: $\{0,1,2\}$ conteo-no-cero solo dice "hay alguien", que ya
lo implica $\{1,2\}$ conteo-no-cero. La aritmética del conteo es la que libera a 0. Y
deducir a alguien limpio es justo lo que luego se puede cosechar como utilidad, así
que esta deducción es la semilla del bienestar extra.""")

code(r"""p = [0.3, 0.3, 0.3]
historia = ((mask_from_indices([0, 1, 2]), 1), (mask_from_indices([1, 2]), 1))
post = bayesian_update_by_counting(p, historia, 3)
print('prior  P(estado activo) :', p)
print('tests              : {0,1,2}=1 , {1,2}=1')
print('posterior augmented :', [round(x, 3) for x in post])
print()
print('x0 = 0.0  -> deducido SANO con certeza (cosechable).')
print('Binario ve {0,1,2}=+ , {1,2}=+ y NO puede concluir el estado de 0.')""")

# ===================================================================
md(r"""## Capa 1b — Por qué la cadena vale para TODA instancia

Conviene aclarar antes de las figuras: la cadena no es un patrón que el código
descubre, es una desigualdad que vale por construcción, y lo empírico es el
**tamaño** de cada brecha.

Cada eslabón añade un poder y nunca puede empeorar el óptimo. Testear de a uno es
un caso particular de planes estáticos sin solapar, que son un caso particular de
estáticos con solape. Un plan estático es un plan dinámico que ignora la historia,
así que el óptimo dinámico no puede ser peor. El test augmented refina al binario
(el binario es el indicador $\mathbf{1}[r>0]$ del conteo), de modo que cualquier
política binaria es ejecutable con conteo, y por eso $U^D \le U^D_A$. Y
$U^{\max} = \sum_i u_i q_i$ es la cota de información total: lo que se obtendría
conociendo el estado de todos. Lo único que medimos abajo es cuánto se separan
estos niveles, que sí depende de la instancia.""")

# ===================================================================
md(r"""## Capa 1c — La jerarquía en instancias concretas: mediana y testigo

Mostramos la escalera en dos instancias, no en una, para no confundir el mecanismo
con la evidencia. La de la izquierda tiene la brecha mediana del conteo; la de la
derecha es una instancia testigo, elegida a propósito por su brecha alta para que el
eslabón $U^D \to U^D_A$ se vea. La diferencia entre ambas ya adelanta que la
ventaja del conteo no es uniforme.""")

code(r"""rng = np.random.default_rng(7)
N, B, G = 5, 3, 5
registros = []
for _ in range(60):
    p = rng.uniform(0.0, 1.0, size=N).tolist()
    u = rng.choice([1.0, 2.0, 3.0], size=N).tolist()
    h = hierarchy_for_instance(p, u, B, G)
    registros.append((h['U_D_A'] - h['U_D'], p, u, h))
registros.sort(key=lambda x: x[0])
mediana = registros[len(registros) // 2]
testigo = registros[-1]
print(f'mediana de la brecha (de 60): {mediana[0]:.4f}')
print(f'testigo (brecha maxima)      : {testigo[0]:.4f}')""")

code(r"""etiquetas = [r'$U^{single}$', r'$U^s_{NO}$', r'$U^s_O$',
             r'$U^D$', r'$U^D_A$', r'$U^{max}$']

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, (gap, p, u, h), titulo in [
        (axes[0], mediana, 'instancia mediana'),
        (axes[1], testigo, 'instancia testigo (brecha alta)')]:
    vals = [h[k] for k in CHAIN]
    colores = ['0.6'] * len(CHAIN); colores[CHAIN.index('U_D_A')] = 'tab:orange'
    ax.bar(range(len(vals)), vals, color=colores, edgecolor='k', linewidth=0.6)
    ax.plot(range(len(vals)), vals, 'k.-', lw=1)
    ax.axhline(h['U_max'], ls=':', color='0.4', lw=1)  # Umax es cota, no peldano
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_title(f'{titulo}\nbrecha U_D_A - U_D = {gap:.3f}', fontsize=10)
    ax.annotate('augmented', xy=(CHAIN.index('U_D_A'), h['U_D_A']),
                xytext=(CHAIN.index('U_D_A') - 0.7, h['U_D_A'] + 0.1),
                color='tab:orange', fontsize=9)
axes[0].set_ylabel('utilidad esperada')
plt.tight_layout(); plt.show()""")

# ===================================================================
md(r"""## Capa 2 — ¿Qué tan seguido es estricto cada eslabón?

Antes de promediar, la objeción honesta: muchas desigualdades son igualdades en una
fracción grande de instancias. La tabla siguiente cuenta, por eslabón, en cuántas
instancias hay separación estricta. En particular, en $N=3$ el conteo no aporta nada
en más de la mitad de los casos; el promedio conteo-no-cero lo cargan unas pocas
instancias.""")

code(r"""df = pd.concat([
    pd.read_csv(os.path.join(ROOT, 'results/hierarchy/hierarchy_small.csv')),
    pd.read_csv(os.path.join(ROOT, 'results/hierarchy/hierarchy_n7.csv')),
], ignore_index=True)
df['benefit_pct'] = (df['U_D_A'] - df['U_D']) / df['U_D'] * 100.0

eslabones = list(zip(CHAIN[:-1], CHAIN[1:]))
filas = []
for N in sorted(df['N'].unique()):
    d = df[df['N'] == N]; n = len(d)
    fila = {'N': N, 'n_inst': n}
    for a, b in eslabones:
        estrictos = ((d[b] - d[a]) > 1e-9).sum()
        fila[f'{a}<{b}'] = f'{estrictos}/{n}'
    filas.append(fila)
print(pd.DataFrame(filas).to_string(index=False))""")

# ===================================================================
md(r"""## Capa 3 — El promedio como cascada (con el techo como cota)

Cada barra registro cuánto bienestar agrega cada régimen respecto al anterior,
etiquetada con el nivel que alcanza: `single`, `static NO`, `static O`, `dynamic`
y `augmented` (es decir $U^{\text{single}} \le \dots \le U^D_A$). La figura compara
óptimos exactos, no el greedy. La barra `augmented` es exactamente el eslabón nuevo
$U^D_A - U^D$, el valor de contar. El techo $U^{\max}$ es una cota superior, no otra
tecnología de testeo, así que va como línea punteada y la cascada termina en
`augmented`; el hueco hasta la línea es el margen que ninguna política, ni la
augmented óptima, puede recuperar.""")

code(r"""N_show = 5
m = df[df['N'] == N_show][CHAIN].mean()
niveles = CHAIN[:-1]  # la cascada llega hasta U_D_A; U_max es cota
incs = [m[niveles[0]]] + [m[niveles[i]] - m[niveles[i-1]]
                          for i in range(1, len(niveles))]
etq = ['single', 'static NO', 'static O', 'dynamic', 'augmented']
colores = ['0.6'] * len(niveles); colores[-1] = 'tab:orange'

fig, ax = plt.subplots(figsize=(8, 4.5))
base = 0.0
for i, inc in enumerate(incs):
    ax.bar(i, inc, bottom=base, color=colores[i], edgecolor='k', linewidth=0.6)
    if i > 0:
        ax.text(i, base + inc + 0.03, f'+{inc:.3f}', ha='center', fontsize=8)
    base += inc
ax.axhline(m['U_max'], ls=':', color='0.4', lw=1.5)
ax.text(len(niveles) - 1, m['U_max'] + 0.03, r'$U^{max}$ (cota de info total)',
        ha='right', color='0.4', fontsize=9)
ax.set_xticks(range(len(niveles))); ax.set_xticklabels(etq, rotation=15)
ax.set_ylabel('utilidad esperada (media)')
ax.set_title(f'Cascada de aportes marginales (N={N_show}, '
             f'{int((df.N==N_show).sum())} instancias)')
plt.tight_layout(); plt.show()
print(f'Aporte medio de CONTAR (U_D_A - U_D) = {incs[-1]:.4f}')""")

# ===================================================================
md(r"""## Capa 4 — El beneficio del conteo: distribución, no solo media

La historia central, pero contada con la distribución completa. Para cada $N$,
el beneficio relativo $(U^D_A - U^D)/U^D$ por instancia, como caja y puntos. La
media sube con la escala, pero la caja revela lo que la media esconde: en $N=3$ la
mediana es cero (el conteo no aporta en la mayoría) y la cola es la que mueve el
promedio; en $N=5$ y $N=7$ la masa ya es conteo-no-cero y dispersa.""")

code(r"""Ns = sorted(df['N'].unique())
datos = [df[df['N'] == N]['benefit_pct'].values for N in Ns]
jr = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.boxplot(datos, positions=Ns, widths=0.6, showfliers=False,
           medianprops=dict(color='k'))
for N, y in zip(Ns, datos):
    x = jr.normal(N, 0.06, size=len(y))
    ax.scatter(x, y, alpha=0.25, s=10, color='tab:orange')
    ax.annotate(f'media {y.mean():.2f}%\nmediana {np.median(y):.2f}%\nn={len(y)}',
                (N, y.max()), textcoords='offset points', xytext=(10, -6),
                fontsize=8)
ax.set_xlabel('N  (con N = G)'); ax.set_xticks(Ns)
ax.set_ylabel(r'beneficio del conteo  $(U^D_A-U^D)/U^D$  [%]')
ax.set_title('Distribución del beneficio por escala (no solo la media)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""**Lectura honesta del crecimiento.** La media sube de $N=3$ a $N=7$, lo cual
es consistente con el mecanismo (el conteo paga vía mejores posteriores sobre un
horizonte más largo). Pero es evidencia, no consulta: son tres puntos, $N$, $B$ y $G$
se mueven a la vez (no aísla cuál causa el crecimiento), y $N=7$ tiene solo 40
instancias. La tendencia es sugerente; el experimento limpio que separe el efecto
de $N$, del horizonte $B$ y del tamaño de pool $G=N$ queda pendiente.""")

# ===================================================================
md(r"""## Capa 5 — La tabla, ya entendida, y las salvedades

Recién ahora la tabla agregada. Se reportan dos formas del beneficio para que no
haya ambigüedad: la media de los cocientes por instancia, $E[(U^D_A-U^D)/U^D]$, que
es la que cuenta la curva de arriba, y el cociente de las medias,
$(\,\overline{U^D_A}-\overline{U^D}\,)/\overline{U^D}$, que no es lo mismo.""")

code(r"""g = df.groupby(['N', 'B', 'G'])
tabla = g[CHAIN].mean()
tabla['E[gap/U_D]_%'] = g['benefit_pct'].mean()
tabla['gapMean/U_Dmean_%'] = (g['U_D_A'].mean() - g['U_D'].mean()) / g['U_D'].mean() * 100
tabla['n_inst'] = g.size()
print(tabla.round(4).to_string())""")

md(r"""**Montaje experimental (sin esconder nada).** Priors $p_i \sim U(0,1)$ y
utilidades $u_i \sim \text{Uniforme}\{1,2,3\}$, instancias independientes. La cadena
$U^{\text{single}} \le \dots \le U^{\max}$ se cumple en todas (cero violaciones).
Configuraciones en la diagonal $N=G$, con $B$ creciendo modestamente. $N=3$ y $N=5$
usan 200 instancias; $N=7$ solo 40, porque el DP exacto $U^D_A$ a $N=7$ es caro
(ramifica en $|t|+1$ resultados), de ahí su caja más rala. Relanzar $N=7$ con más
instancias está listo en el código.""")

# ===================================================================
md(r"""## Preguntas para Francisco

Tres ganchos. ¿El valor por paso bajo conteo es submodular adaptativo? De serlo, el
greedy heredaría una garantía $1 - 1/e$. ¿Cómo cambia la separación cuando el conteo
se observa con ruido, como el cycle threshold real de la counting? ¿En qué clases de
diseños (disjuntos, laminares, treewidth acotado) la inferencia exacta es tratable y
la separación se puede caracterizar en cerrado?""")

nbf.write(nb, OUT)
print(f'wrote {OUT} ({len(nb.cells)} cells)')
