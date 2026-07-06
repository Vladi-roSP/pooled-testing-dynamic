"""
Bitmask helpers and core primitives for DAPTS.

Individuals are indexed 0..n-1.  A "mask" is a Python int whose bit i
is set iff individual i is in the set (pool, latent-state profile, etc.).
"""

from itertools import combinations


def mask_from_indices(indices):
    """Bitmask with the given bit positions set. e.g. [0,2] -> 0b101."""
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def indices_from_mask(mask, n=None):
    """Sorted list of set-bit positions. If n given, ignore bits >= n."""
    if n is not None:
        mask &= (1 << n) - 1
    out = []
    while mask:
        lsb = mask & -mask
        out.append(lsb.bit_length() - 1)
        mask ^= lsb
    return out


def popcount(mask):
    """Number of set bits."""
    return mask.bit_count()


def mask_str(mask, n=None):
    """Human-readable string for a bitmask. e.g. 0b101 -> '{0,2}'."""
    idxs = indices_from_mask(mask, n)
    return "{" + ",".join(str(i) for i in idxs) + "}" if idxs else "{}"


def all_pools(n, G, include_empty=True):
    """Every pool mask t with |t| <= G over population 0..n-1."""
    pools = []
    start = 0 if include_empty else 1
    for size in range(start, min(G, n) + 1):
        for combo in combinations(range(n), size):
            pools.append(mask_from_indices(combo))
    return pools


def compute_active_mask(p, cleared_mask, n, threshold=1e-10,
                        include_known_clearancey=False):
    """Bitmask of individuals whose latent_state status is still uncertain.

    An individual is *inactive* (excluded from future pools) if:
      - already cleared (bit set in cleared_mask), or
      - p_i <= threshold  (essentially known clearancey), or
      - p_i >= 1 - threshold (essentially confirmed active).

    Returns (active_mask, confirmed_active_mask).

    Parameters
    ----------
    include_known_clearancey : bool
        If True, individuals deduced clearancey (p_i <= threshold) but not yet
        CLEARED are kept in the active mask. This matters because utility is
        only credited when an individual is physically placed in a pool that
        returns r=0 (see the welfare objective): a deduced-clearancey individual
        carries zero risk (it never raises the count) yet still owns utility
        u_i that can only be harvested by counting it in a guaranteed-r=0 pool.
        Filtering it out forfeits u_i forever — a genuine sub-optimality the
        DP avoids. Greedy pool-selectors pass True; the default stays False to
        preserve the contract for other callers.
    """
    active = 0
    confirmed_active = 0
    for i in range(n):
        if cleared_mask >> i & 1:
            continue  # already cleared — utility already collected
        if p[i] <= threshold:
            if include_known_clearancey:
                active |= 1 << i  # zero-risk; still harvestable for its u_i
            continue  # known clearancey
        if p[i] >= 1.0 - threshold:
            confirmed_active |= 1 << i
            continue  # confirmed active — would force r>=1, never cleared
        active |= 1 << i
    return active, confirmed_active


def all_pools_from_mask(active_mask, G, include_empty=True):
    """Generate pool masks using only individuals in active_mask.

    Like all_pools but restricted to the set bits of active_mask.
    """
    active_indices = indices_from_mask(active_mask)
    m = len(active_indices)
    pools = []
    start = 0 if include_empty else 1
    for size in range(start, min(G, m) + 1):
        for combo in combinations(active_indices, size):
            pools.append(mask_from_indices(combo))
    return pools


def test_result(pool_mask, z_mask):
    """Augmented test result r(t, Z) = |t ∩ Z| (count of active in pool)."""
    return popcount(pool_mask & z_mask)


def bin_of(r, cap):
    """Cuantizador de truncamiento del conteo r a un bin.

    cap is None  -> conteo completo (identidad): devuelve r.
    cap (int)    -> min(r, cap). Con cap >= 1 el bin 0 es exactamente {0}
                    (aísla {0}); cap = 1 es el régimen binario (0 vs >=1).
    """
    return r if cap is None else min(r, cap)
