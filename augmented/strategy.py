"""
DAPTS strategy representation.

A DAPTS F = (F1, ..., FB) maps histories to pools.
  Fk : H_{k-1} -> P_G([n])
  History h_k = tuple of k (pool_mask, result) pairs.
"""

# A History is a tuple of (pool_mask, test_result) pairs.
History = tuple  # e.g. ((pool1, r1), (pool2, r2), ...)


class DAPTS:
    """Dynamic Augmented Adaptive Group Counting Strategy.

    policy[k] maps a History of length k to the pool chosen at step k+1.
    So policy[0] maps () -> pool for the first test, etc.

    cap is the truncation quantizer the policy was solved under (see
    core.bin_of): the outcomes recorded in each history key are binned
    counts min(r, cap), not raw counts. cap=None means full counts, so the
    binned outcome equals the raw count and any consumer keyed on raw counts
    stays correct. For cap<G the outcomes are coarsened, so replayers must
    bin the observed count through the same quantizer before looking a
    history up.
    """

    def __init__(self, B, policy=None, cap=None):
        self.B = B
        self.policy = policy if policy is not None else [{} for _ in range(B)]
        self.cap = cap

    def choose(self, k, history):
        """Pool mask for step k (1-indexed) given history of length k-1."""
        return self.policy[k - 1][history]

    def set_action(self, k, history, pool_mask):
        """Record: at step k with this history, choose pool_mask."""
        self.policy[k - 1][history] = pool_mask

    def __repr__(self):
        entries = sum(len(d) for d in self.policy)
        return f"DAPTS(B={self.B}, entries={entries})"
