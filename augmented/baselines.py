"""
Baseline benchmarks from Section 2.1.

U_max    — upper bound: sum of u_i * q_i (infinite budget).
U_single — optimal strategy counting one individual per test.
"""


def u_max(p, u):
    """U^max = sum_i u_i * q_i.  Upper bound on expected utility."""
    return sum(ui * (1.0 - pi) for ui, pi in zip(u, p))


def u_single(p, u, B):
    """U^single = sum of top min(B, n) values of u_i * q_i.

    Returns (value, selected_indices).
    """
    scores = sorted([(u[i] * (1.0 - p[i]), i) for i in range(len(p))],
                    reverse=True)
    k = min(B, len(p))
    selected = [idx for _, idx in scores[:k]]
    value = sum(s for s, _ in scores[:k])
    return value, selected
