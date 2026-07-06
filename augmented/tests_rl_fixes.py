"""
Regression tests for the DaptsBucketEnv fixes (audit 2026-06-09):
  * the observation could leave the declared Box (usum can exceed N);
  * utility_bin_edges was hardcoded [0,2,3], ignoring the utility_bins arg.

Run with:  PYTHONPATH=. python augmented/tests_rl_fixes.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from augmented.rl_env import DaptsBucketEnv


def test_bucket_obs_stays_in_box_when_usum_exceeds_N():
    # N=3, G=3, utilities all 3 -> usum can reach 9 > N=3. The observation must
    # remain inside the declared observation_space.
    def generator(rng):
        return [0.3, 0.3, 0.3], [3.0, 3.0, 3.0]
    env = DaptsBucketEnv(generator, B=2, G=3, N=3)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset obs outside Box"
    done = False
    steps = 0
    while not done and steps < 10:
        # add an agent from whichever category is populated; else STOP
        action = next((c for c in range(env.num_categories)
                       if env.category_counts.get(c, 0) > 0), env.num_categories)
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
        assert env.observation_space.contains(obs), \
            f"step obs outside Box: usum={obs[-2]}, high={env.observation_space.high[-2]}"


def test_utility_bins_param_is_honored():
    # With utility_bins=4 the binning must be able to produce 4 distinct bins,
    # i.e. _category's utility component spans 0..3 (not capped at 3 fixed edges).
    def generator(rng):
        return [0.3] * 8, list(range(8))
    env = DaptsBucketEnv(generator, B=2, G=3, N=8, utility_bins=4)
    # Number of interior+endpoint structure must yield exactly utility_bins bins.
    bins_seen = set()
    for u in np.linspace(0, env.utility_high, 50):
        ub = int(np.digitize(u, env.utility_bin_edges) - 1)
        ub = min(max(ub, 0), env.utility_bins - 1)
        bins_seen.add(ub)
    assert bins_seen == set(range(env.utility_bins)), \
        f"utility_bins=4 but only bins {sorted(bins_seen)} reachable"


def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and getattr(v, "__module__", None) == __name__]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
