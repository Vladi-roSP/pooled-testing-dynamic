"""Registro de funciones de valor aproximado V-hat para la cota penalizada.

Este modulo es la SUPERFICIE EDITABLE de la busqueda de certificados (el
harness dapts-autoresearch solo puede tocar este archivo). El contrato que lo
hace seguro: la penalizacion de certificates.py es una diferencia de
martingala de V-hat bajo la filtracion natural, asi que CUALQUIER funcion
registrada aqui produce una cota superior valida del optimo (teorema de
relajacion de informacion, Brown-Smith-Sun). Editar V-hat solo puede cambiar
que tan APRETADA es la cota, nunca su validez — que ademas se verifica
instancia por instancia en el benchmark y en tests_certificates.py.

Interfaz: una V-hat es fn(ctx, h_fs, remaining) -> float, donde
  ctx        expone el problema y primitivas cacheadas Y ESCALABLES:
             ctx.p, ctx.u, ctx.n, ctx.G,
             ctx.posterior(h_fs)   -> [P(Z_i=1 | h)]  (marginales; exacto en n
                                      chico, Gibbs por componentes en n grande)
             ctx.cleared_mask(h_fs)-> bitmask de acreditados (pools con r=0)
             ctx.greedy_value(p, u, budget) -> EU del greedy miope
  h_fs       la historia como frozenset de pares (pool_mask, r)
  remaining  tests que quedan DESPUES de observar el resultado de este paso

REGLA DURA DE ESCALABILIDAD (jul 2026). Una V-hat debe correr en tiempo que
NO crezca como resolver el problema. En concreto: NO enumerar el soporte
CONJUNTO del posterior (nada de `for z in range(1 << n)` ni listas de tamano
2^n), NO llamar al solver exacto (solve_optimal_dapts) ni recomputar el value
function optimo. El benchmark aplica una prueba de escalabilidad (llama a la
V-hat en una instancia n~32 con tope de tiempo); una V-hat que enumera el
conjunto o resuelve el problema TRUENA ahi y la corrida se descarta.
Construyela SOLO con las primitivas de ctx (marginales, greedy_value), que ya
escalan. Motivo: certificar importa en n=50, donde el value function exacto es
justo lo intratable — usarlo da una cota valida pero circular (apretada porque
resolvio el problema), inutil para la mision.

Hallazgos que orientan el diseno (jul 2026): la V-hat buena es insesgada
antes que precisa (el adversario interno explota sesgos, p.ej. el error de
independencia del greedy-a-futuro), y su alcance debe crecer con el horizonte
(el apriete de la V-hat miope muere en B=3). La direccion abierta es una V-hat
con profundidad d(B) construida sobre marginales/rollouts, no sobre el
soporte conjunto.
"""

VHAT_REGISTRY = {}


def register(name):
    """Decorador: registra una V-hat bajo `name` para usarla como
    u_pen_exact(..., v_hat=name)."""
    def deco(fn):
        VHAT_REGISTRY[name] = fn
        return fn
    return deco


def get(name):
    try:
        return VHAT_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(VHAT_REGISTRY))
        raise KeyError(f"V-hat desconocida: {name!r}. Registradas: {known}")


@register("zero")
def vhat_zero(ctx, h_fs, remaining):
    """V-hat nula: penalizacion cero; recupera exactamente U_PI."""
    return 0.0


@register("umax")
def vhat_umax(ctx, h_fs, remaining):
    """Potencial posterior sum_i u_i * P(Z_i=0 | h). Lineal en las marginales
    exactas (insesgado donde el adversario mira). El mejor conocido a la
    fecha: aprieta +4 a +5 puntos en B=2 y nada en B=3."""
    post = ctx.posterior(h_fs)
    return sum(ctx.u[i] * (1.0 - post[i]) for i in range(ctx.n))


@register("greedy")
def vhat_greedy(ctx, h_fs, remaining):
    """Valor-a-futuro del greedy miope jugando `remaining` tests desde el
    posterior de h, con lo ya acreditado puesto a cero. Documentado como MAS
    FLOJO que umax: alimenta marginales al greedy como si fueran priors
    independientes y el adversario explota ese sesgo (ver
    test_u_pen_vhat_comparison_documented)."""
    if remaining <= 0:
        return 0.0
    post = ctx.posterior(h_fs)
    cleared = ctx.cleared_mask(h_fs)
    u_rem = [0.0 if (cleared >> i & 1) else ctx.u[i] for i in range(ctx.n)]
    return ctx.greedy_value(post, u_rem, remaining)


@register("research")
def vhat_research(ctx, h_fs, remaining):
    """El slot de busqueda del harness de autoresearch: el benchmark de
    certificados siempre evalua esta V-hat. Arranca como copia de umax (el
    mejor conocido); el loop de investigacion la reemplaza por candidatas
    mejores y conserva solo lo que aprieta el certificado."""
    post = ctx.posterior(h_fs)
    return sum(ctx.u[i] * (1.0 - post[i]) for i in range(ctx.n))
