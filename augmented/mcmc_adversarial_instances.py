"""Familias de instancias adversariales para la misión de ablación MCMC.

Cada instancia es {"name", "n", "p", "history"}; history es una tupla de
pares (members, r) con members la tupla de agentes del pool y r el conteo
exacto. El benchmark evalúa la familia registrada como "research".

Reglas (program.md de dapts-autoresearch): n <= 16 (la enumeración exacta es
la referencia), fibra con >= 2 niveles de conteo, y la instancia debe forzar
el camino MCMC. El objetivo es maximizar el sesgo del kernel swap-only
(count-preserving) manteniendo el kernel completo dentro de tolerancia.
"""

# Instancia canónica del fix de ergodicidad: fibra {(0,1,0), (1,0,1)} con
# niveles de conteo {1, 2}; exacto [0.15, 0.85, 0.15].
CANONICAL = {
    "name": "cadena_impar_n3",
    "n": 3,
    "p": [0.15, 0.15, 0.15],
    "history": (((0, 1), 1), ((1, 2), 1)),
}

# Cadena impar más larga: x_{i+1} = 1 - x_i fuerza dos perfiles complementarios
# {(1,0,1,0,1,0,1), (0,1,0,1,0,1,0)} con niveles {4, 3}. El único movimiento
# que conecta ambos voltea los siete agentes a la vez (camino abierto,
# cambia el conteo), así que el kernel swap-only queda atrapado. Las cadenas
# pares ponen ambos perfiles en el mismo nivel y caen en el gate de un nivel.
ODD_CHAIN_7 = {
    "name": "cadena_impar_n7",
    "n": 7,
    "p": [0.15] * 7,
    "history": tuple(((i, i + 1), 1) for i in range(6)),
}

# Triples solapados (familia del audit de Hastings, 2026-07-06) con priors
# heterogéneos: fibra con niveles {1, 2} (el agente compartido activo, o uno
# activo por triple).
TRIPLE_OVERLAP = {
    "name": "triples_solapados_n5",
    "n": 5,
    "p": [0.35, 0.06, 0.14, 0.23, 0.28],
    "history": (((0, 1, 2), 1), ((2, 3, 4), 1)),
}

FAMILIES = {
    "research": [CANONICAL, ODD_CHAIN_7, TRIPLE_OVERLAP],
}
