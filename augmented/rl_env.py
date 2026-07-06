"""
Gymnasium environments for augmented DAPTS, ready for reinforcement learning.

Both environments are powered entirely by the ``augmented`` package (Bayesian
updates and myopic-greedy pool selection) -- there is NO MOSEK / conic-solver
dependency, unlike the original ``classical/rl_training`` code.

  DaptsExactEnv   Exact belief-state MDP for SMALL n. The observation is the
                  full posterior over the 2**n latent-state profiles, so a trained
                  policy can in principle match the DP optimum
                  (augmented.solver.solve_optimal_dapts). Use it to VALIDATE
                  that RL recovers the optimum.

  DaptsBucketEnv  Bucketed environment for LARGE N (beyond the n<=14 wall of
                  exact DP). It is a port of
                  ``classical/rl_training/PPO_bucket_gymnasium_B*.py``: the
                  observation is a fixed-size histogram of agents by
                  (clearance bucket, utility bucket). The RL agent chooses the
                  FIRST pool; the remaining B-1 tests are played by augmented
                  myopic greedy.

Reproducibility
---------------
Every stochastic choice goes through the env's gymnasium ``np_random``, seeded
via ``reset(seed=...)``. Instance generators receive that same Generator, so a
fixed seed fully determines an episode (instance, true profile, shuffles).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from augmented.core import all_pools, indices_from_mask, mask_from_indices, test_result
from augmented.bayesian import bayesian_update_single_test
from augmented.greedy import _myopic_best_pool, greedy_myopic_simulate


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def prior_profile_weights(p):
    """Return w[z] = Pr(Z = z) under the independent prior, for all 2**n z."""
    n = len(p)
    w = np.ones(1 << n, dtype=np.float64)
    for z in range(1 << n):
        prob = 1.0
        for i in range(n):
            prob *= p[i] if (z >> i) & 1 else (1.0 - p[i])
        w[z] = prob
    return w


def draw_profile(p, rng):
    """Draw a true latent-state profile z (bitmask) from independent prior p."""
    z = 0
    for i in range(len(p)):
        if rng.random() < p[i]:
            z |= 1 << i
    return z


# -------------------------------------------------------------------
# Exact belief-state environment (validation against DP)
# -------------------------------------------------------------------

class DaptsExactEnv(gym.Env):
    """Exact belief-MDP for augmented DAPTS on small n.

    State (observation): the full posterior over the 2**n profiles, plus a
    cleared-individuals indicator and the fraction of budget remaining. This
    is the *sufficient statistic* of the problem, so an optimal policy on this
    env attains the DP optimum.

    Parameters
    ----------
    instance_generator : callable
        ``instance_generator(rng) -> (p, u)`` where p, u are length-n sequences.
        For a fixed instance, return the same (p, u) ignoring rng.
    B, G, n : int
        Budget (tests), pool size, population size.
    max_n : int
        Guard rail; the observation has 2**n entries.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance_generator, B, G, n, max_n=10):
        super().__init__()
        if n > max_n:
            raise ValueError(
                f"DaptsExactEnv is for small n (got n={n} > max_n={max_n}); "
                "use DaptsBucketEnv to scale.")
        self.instance_generator = instance_generator
        self.B, self.G, self.n = B, G, n
        self.num_profiles = 1 << n
        self.pools = all_pools(n, G, include_empty=False)

        self.action_space = spaces.Discrete(len(self.pools))
        obs_dim = self.num_profiles + n + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.p = self.u = None
        self._belief = None
        self._cleared = 0
        self._k = 0
        self._z = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p, u = self.instance_generator(self.np_random)
        self.p, self.u = list(p), list(u)
        if len(self.p) != self.n:
            raise ValueError(
                f"instance_generator produced n={len(self.p)}, env expects {self.n}")

        self._belief = prior_profile_weights(self.p)
        # The true profile may be forced (used by the exact evaluator).
        if options is not None and "force_z" in options:
            self._z = int(options["force_z"])
        else:
            self._z = int(self.np_random.choice(self.num_profiles,
                                                p=self._belief))
        self._cleared = 0
        self._k = 0
        return self._obs(), {}

    def _obs(self):
        cleared = np.array([(self._cleared >> i) & 1 for i in range(self.n)],
                           dtype=np.float32)
        remaining = np.array([1.0 - self._k / self.B], dtype=np.float32)
        return np.concatenate(
            [self._belief.astype(np.float32), cleared, remaining])

    def step(self, action):
        pool = self.pools[int(action)]
        r = test_result(pool, self._z)

        # Posterior update: keep only profiles consistent with observed r.
        keep = np.fromiter(
            (1.0 if test_result(pool, z) == r else 0.0
             for z in range(self.num_profiles)),
            dtype=np.float64, count=self.num_profiles)
        self._belief = self._belief * keep
        total = self._belief.sum()
        if total > 0:
            self._belief /= total

        if r == 0:
            self._cleared |= pool
        self._k += 1

        done = self._k >= self.B
        reward = 0.0
        if done:
            reward = float(sum(self.u[i] for i in range(self.n)
                               if (self._cleared >> i) & 1))
        return self._obs(), reward, done, False, {}


# -------------------------------------------------------------------
# Bucketed environment (scaling beyond the exact-DP wall)
# -------------------------------------------------------------------

class DaptsBucketEnv(gym.Env):
    """Bucketed environment for large populations.

    Port of ``classical/rl_training/PPO_bucket_gymnasium_B*.py`` with the
    conic/MOSEK machinery replaced by augmented myopic greedy:

      * Observation: histogram of the N agents over
        (clearance bucket x utility bucket), plus the running utility sum and
        clearance product of the agents picked so far. Fixed size -> scales in N.
      * Action: pick an agent from a (clearance, utility) category, or STOP.
        The RL agent assembles the FIRST pool (up to G agents).
      * Reward: a true profile z is drawn; test 1 is the RL pool; tests
        2..B are played by augmented myopic greedy. Reward = utility of all
        individuals proven clearancey.

    Parameters
    ----------
    instance_generator : callable
        ``instance_generator(rng) -> (p, u)`` with length-N sequences;
        p[i] = latent-state probability, u[i] = utility.
    B, G, N : int
        Budget, pool size, population size.
    clearance_bins, utility_bins : int
        Discretisation of the (clearance, utility) observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance_generator, B, G, N,
                 clearance_bins=4, utility_bins=3, utility_high=3.0):
        super().__init__()
        self.instance_generator = instance_generator
        self.B, self.G, self.N = B, G, N
        self.clearance_bins = clearance_bins
        self.utility_bins = utility_bins
        self.utility_high = float(utility_high)
        self.num_categories = clearance_bins * utility_bins

        # Per-dimension bounds: the first num_categories entries are agent counts
        # (<= N); the last two are usum (sum of selected utilities, which can
        # reach G*utility_high and so EXCEED N) and hprod in [0,1]. Declaring a
        # flat high=N truncated the observation, silently corrupting the state.
        high = np.empty(self.num_categories + 2, dtype=np.float32)
        high[:self.num_categories] = float(N)
        high[self.num_categories] = float(G) * self.utility_high  # usum
        high[self.num_categories + 1] = 1.0                       # hprod
        self.observation_space = spaces.Box(
            low=0.0, high=high,
            shape=(self.num_categories + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_categories + 1)

        self.clearance_bin_edges = np.linspace(0.0, 1.0, clearance_bins + 1)
        # Full edges of a uniform partition of [0, utility_high] into exactly
        # utility_bins bins (same convention as clearance_bin_edges, consumed by
        # _category via digitize-1). Honors the utility_bins arg instead of the
        # old hardcoded [0,2,3] that silently fixed it at 3 bins over {1,2,3}.
        self.utility_bin_edges = np.linspace(0.0, self.utility_high,
                                             utility_bins + 1)

        self.p = self.u = None
        self.category_agents = {}
        self.category_counts = {}
        self.selected = []
        self.attempts = 0
        self.last_z = 0          # true profile of the most recent episode

    def _category(self, clearance, utility):
        ub = int(np.digitize(utility, self.utility_bin_edges) - 1)
        ub = min(max(ub, 0), self.utility_bins - 1)
        hb = int(np.digitize(clearance, self.clearance_bin_edges) - 1)
        hb = min(max(hb, 0), self.clearance_bins - 1)
        return ub * self.clearance_bins + hb

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p, u = self.instance_generator(self.np_random)
        self.p = np.asarray(p, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float64)

        self.category_agents = {c: [] for c in range(self.num_categories)}
        clearance = 1.0 - self.p
        for i in range(self.N):
            cat = self._category(clearance[i], self.u[i])
            self.category_agents[cat].append(i)
        for c in self.category_agents:
            self.np_random.shuffle(self.category_agents[c])
        self.category_counts = {c: len(self.category_agents[c])
                                for c in range(self.num_categories)}
        self.selected = []
        self.attempts = 0
        return self._obs(), {}

    def _obs(self):
        counts = np.array([self.category_counts[c]
                           for c in range(self.num_categories)],
                          dtype=np.float32)
        usum = float(sum(self.u[i] for i in self.selected))
        hprod = (float(np.prod([1.0 - self.p[i] for i in self.selected]))
                 if self.selected else 1.0)
        return np.concatenate([counts, [usum, hprod]]).astype(np.float32)

    def step(self, action):
        action = int(action)
        done = False
        if action == self.num_categories:           # STOP
            done = True
        elif self.category_counts.get(action, 0) > 0:
            agent = self.category_agents[action].pop()
            self.category_counts[action] -= 1
            self.selected.append(agent)

        self.attempts += 1
        if self.attempts >= self.G:
            done = True

        reward = 0.0
        if done:
            reward = self._rollout_reward()
        return self._obs(), reward, done, False, {}

    def _rollout_reward(self):
        """Draw z, play test 1 = RL pool, tests 2..B = augmented greedy."""
        z = draw_profile(self.p, self.np_random)
        self.last_z = z
        rl_pool = mask_from_indices(self.selected) if self.selected else 0

        calls = {"k": 0}

        def selector(p, u, G, n, cleared_mask):
            first = calls["k"] == 0
            calls["k"] += 1
            if first and rl_pool != 0:
                return rl_pool
            return _myopic_best_pool(p, u, G, n, cleared_mask)

        _, _, util = greedy_myopic_simulate(
            list(self.p), list(self.u), self.B, self.G, z,
            pool_selector=selector)
        return float(util)
