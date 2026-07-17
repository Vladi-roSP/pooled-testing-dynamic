"""
Solver-based optimal pool selection for large-n instances.

Two backends:
  mosek_best_pool()  — exponential cone (exact via Mosek Fusion API)
  gurobi_best_pool() — MILP with log general constraint (Gurobi)

Both return a pool bitmask, same interface as greedy._myopic_best_pool().
Signature: (p, u, G, n, cleared_mask) -> int
"""

import math
import warnings

from augmented.core import (
    mask_from_indices, indices_from_mask, compute_active_mask,
)

# Backend que realmente produjo el ultimo pool devuelto por
# mosek_best_pool / gurobi_best_pool: "mosek", "gurobi" o "heuristic".
# Los numeros publicados dependen de esto; nunca mas un fallback silencioso.
LAST_BACKEND = None


def _heuristic_best_pool(active_indices, p, u, G):
    """Fallback: pick G individuals with highest u_i * q_i."""
    scored = [(u[i] * (1.0 - p[i]), i) for i in active_indices]
    scored.sort(reverse=True)
    selected = [idx for _, idx in scored[:G]]
    return mask_from_indices(selected)


def mosek_best_pool(p, u, G, n, cleared_mask):
    """Find optimal myopic pool using Mosek exponential cone.

    Solves:
        max  y + Σ x_i log(q_i)
        s.t. z = Σ u_i x_i
             z ≥ ε
             (z, 1, y) ∈ K_exp   [y ≤ log(z)]
             Σ x_i ≤ G
             Σ x_i ≥ 1
             x_i ∈ {0,1}

    Returns pool bitmask (0 if no useful pool).
    """
    global LAST_BACKEND
    # Identify active individuals
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        return 0

    # Extract active probabilities and utilities
    active_p = [p[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    # Check: all utilities zero → no useful pool
    if all(ui <= 0 for ui in active_u):
        return 0

    # Compute log(q_i) coefficients
    log_q = []
    for pi in active_p:
        qi = 1.0 - pi
        log_q.append(math.log(qi) if qi > 1e-15 else -35.0)

    EPS = 1e-8

    try:
        # Import inside the try so a missing backend / license falls back to the
        # heuristic instead of raising (active_indices is already computed above).
        from mosek.fusion import (
            Model, Domain, Expr, ObjectiveSense, Var, SolutionStatus,
        )
        with Model('mosek_pool') as M:
            M.setSolverParam("log", "0")
            M.setSolverParam("mioMaxTime", 30.0)
            M.setSolverParam("mioTolRelGap", 1e-3)

            # Variables
            x = M.variable("x", n_active, Domain.binary())
            y = M.variable("y", 1, Domain.unbounded())
            z = M.variable("z", 1, Domain.greaterThan(EPS))
            d = M.variable("d", 1, Domain.equalsTo(1.0))

            # Exponential cone: (z, d, y) ∈ K_exp → y ≤ log(z)
            t = Var.vstack(z.index(0), d.index(0), y.index(0))
            M.constraint("expc", t, Domain.inPExpCone())

            # z = Σ u_i x_i
            M.constraint("util",
                          Expr.sub(Expr.dot(active_u, x), z.index(0)),
                          Domain.equalsTo(0.0))

            # Pool size constraints
            M.constraint("pool_max", Expr.sum(x),
                          Domain.lessThan(float(G)))
            M.constraint("pool_min", Expr.sum(x),
                          Domain.greaterThan(1.0))

            # Objective: max y + Σ x_i log(q_i)
            M.objective("obj", ObjectiveSense.Maximize,
                         Expr.add(y.index(0), Expr.dot(log_q, x)))

            M.solve()

            # Check solution status
            sol_status = M.getPrimalSolutionStatus()
            if sol_status not in (SolutionStatus.Optimal,
                                  SolutionStatus.Feasible):
                LAST_BACKEND = "heuristic"
                return _heuristic_best_pool(active_indices, p, u, G)

            # Extract solution
            x_vals = x.level()
            selected = [active_indices[i] for i in range(n_active)
                        if x_vals[i] > 0.5]

            if not selected:
                LAST_BACKEND = "heuristic"
                return _heuristic_best_pool(active_indices, p, u, G)

            LAST_BACKEND = "mosek"
            return mask_from_indices(selected)

    except Exception as e:
        LAST_BACKEND = "heuristic"
        warnings.warn(
            f"mosek no disponible ({e}): usando _heuristic_best_pool "
            "(calidad inferior)",
            RuntimeWarning, stacklevel=2,
        )
        return _heuristic_best_pool(active_indices, p, u, G)


def gurobi_best_pool(p, u, G, n, cleared_mask):
    """Find optimal myopic pool using Gurobi MILP with log constraint.

    Solves:
        max  y + Σ x_i log(q_i)
        s.t. z = Σ u_i x_i
             y = log(z)     [general constraint]
             Σ x_i ≤ G
             Σ x_i ≥ 1
             x_i ∈ {0,1}
             z ≥ ε

    Returns pool bitmask (0 if no useful pool).
    """
    global LAST_BACKEND
    # Identify active individuals
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    active_indices = indices_from_mask(active_mask, n)
    n_active = len(active_indices)

    if n_active == 0:
        return 0

    active_p = [p[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    if all(ui <= 0 for ui in active_u):
        return 0

    log_q = []
    for pi in active_p:
        qi = 1.0 - pi
        log_q.append(math.log(qi) if qi > 1e-15 else -35.0)

    EPS = 1e-8
    U_max = G * max(active_u)

    try:
        # Import inside the try so a missing backend / license falls back to the
        # heuristic instead of raising (active_indices is already computed above).
        import gurobipy as gp
        from gurobipy import GRB
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()

            with gp.Model('gurobi_pool', env=env) as m:
                m.setParam('TimeLimit', 30.0)
                m.setParam('MIPGap', 1e-3)

                # Variables
                x = m.addVars(n_active, vtype=GRB.BINARY, name='x')
                z = m.addVar(lb=EPS, ub=U_max, name='z')
                y = m.addVar(lb=-GRB.INFINITY, name='y')

                # z = Σ u_i x_i
                m.addConstr(
                    z == gp.quicksum(active_u[i] * x[i]
                                     for i in range(n_active)),
                    name='util',
                )

                # Pool size constraints
                m.addConstr(
                    gp.quicksum(x[i] for i in range(n_active)) <= G,
                    name='pool_max',
                )
                m.addConstr(
                    gp.quicksum(x[i] for i in range(n_active)) >= 1,
                    name='pool_min',
                )

                # y = log(z) via general constraint
                m.addGenConstrLog(z, y, name='log_z')

                # Objective: max y + Σ x_i log(q_i)
                m.setObjective(
                    y + gp.quicksum(log_q[i] * x[i]
                                    for i in range(n_active)),
                    GRB.MAXIMIZE,
                )

                m.optimize()

                # Check solution status (handle OPTIMAL, SUBOPTIMAL,
                # and TIME_LIMIT with incumbent)
                has_solution = (
                    m.Status == GRB.OPTIMAL
                    or (m.Status in (GRB.SUBOPTIMAL, GRB.TIME_LIMIT)
                        and m.SolCount > 0)
                )

                if not has_solution:
                    LAST_BACKEND = "heuristic"
                    return _heuristic_best_pool(active_indices, p, u, G)

                selected = [active_indices[i] for i in range(n_active)
                            if x[i].X > 0.5]

                if not selected:
                    LAST_BACKEND = "heuristic"
                    return _heuristic_best_pool(active_indices, p, u, G)

                LAST_BACKEND = "gurobi"
                return mask_from_indices(selected)

    except Exception as e:
        LAST_BACKEND = "heuristic"
        warnings.warn(
            f"gurobi no disponible ({e}): usando _heuristic_best_pool "
            "(calidad inferior)",
            RuntimeWarning, stacklevel=2,
        )
        return _heuristic_best_pool(active_indices, p, u, G)


def sample_best_pool(draws, u, G, n, cleared_mask):
    """Seleccion de pool por frecuencia conjunta muestral.

    score(t) = (#draws con t completamente limpio / S) * sum(u_i, i en t no
    acreditado). Construccion greedy: en cada paso agrega el miembro que
    maximiza el score del pool extendido; se detiene si ningun miembro lo
    mejora. Mata el sesgo de independencia del producto de marginales
    (los draws ya cargan las correlaciones del posterior).
    """
    S = len(draws)
    if S == 0:
        return 0
    candidates = [i for i in range(n) if not (cleared_mask >> i & 1)]
    pool, alive, gain, best_score = 0, list(range(S)), 0.0, 0.0
    for _ in range(G):
        best = None
        for i in candidates:
            if pool >> i & 1:
                continue
            bit = 1 << i
            alive_i = [k for k in alive if not (draws[k] & bit)]
            g = gain + (u[i] if not (cleared_mask >> i & 1) else 0.0)
            score = (len(alive_i) / S) * g
            if score > best_score + 1e-15:
                best, best_score, best_alive, best_gain = i, score, alive_i, g
        if best is None:
            break
        pool |= (1 << best)
        alive, gain = best_alive, best_gain
    return pool


def solver_best_pool(p, u, G, n, cleared_mask, solver='mosek'):
    """Dispatch to Mosek or Gurobi based on solver parameter.

    Parameters
    ----------
    solver : str
        'mosek' or 'gurobi'.

    Returns pool bitmask.
    """
    if solver == 'mosek':
        return mosek_best_pool(p, u, G, n, cleared_mask)
    elif solver == 'gurobi':
        return gurobi_best_pool(p, u, G, n, cleared_mask)
    else:
        raise ValueError(f"Unknown solver: {solver!r}. Use 'mosek' or 'gurobi'.")
