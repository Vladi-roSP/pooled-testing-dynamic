"""
Unit tests for state_reward_greedy.

Run with: python augmented/tests_state_reward_greedy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import mask_from_indices
from augmented.greedy import greedy_myopic_expected_utility
from augmented.state_reward_greedy import (
    _compute_info_gain,
    _beta_best_pool,
    greedy_myopic_beta_expected_utility,
    greedy_myopic_beta_simulate,
)


def test_info_gain_entropy_nonzero_count():
    p = [0.2, 0.3, 0.1]
    pool = mask_from_indices([0, 1])
    info_gain = _compute_info_gain(
        pool, p, n=3, info_metric='entropy', cleared_mask=0)
    assert info_gain >= -1e-10


def test_info_gain_confirmed_zero_for_deterministic_pool():
    p = [0.0, 0.0, 0.4]
    pool = mask_from_indices([0, 1])
    info_gain = _compute_info_gain(
        pool, p, n=3, info_metric='confirmed', cleared_mask=0)
    assert abs(info_gain) < 1e-10


def test_beta_zero_matches_standard_greedy_eu():
    p = [0.1, 0.2, 0.15]
    u = [5.0, 3.0, 4.0]
    eu_standard = greedy_myopic_expected_utility(p, u, B=2, G=2)
    eu_beta = greedy_myopic_beta_expected_utility(
        p, u, B=2, G=2, beta=0.0, info_metric='entropy')
    assert abs(eu_standard - eu_beta) < 1e-10


def test_beta_zero_matches_standard_first_pool():
    p = [0.1, 0.2, 0.15]
    u = [5.0, 3.0, 4.0]
    pool = _beta_best_pool(p, u, G=2, n=3, cleared_mask=0,
                           beta=0.0, info_metric='entropy')
    assert pool == mask_from_indices([0, 2])


def test_beta_simulate_returns_valid_history():
    p = [0.1, 0.2, 0.15]
    u = [5.0, 3.0, 4.0]
    history, cleared_mask, utility = greedy_myopic_beta_simulate(
        p, u, B=2, G=2, z_mask=0, beta=1.0, info_metric='variance')
    assert len(history) <= 2
    assert cleared_mask >= 0
    assert utility >= 0


def _run_all():
    import inspect
    this = inspect.getmodule(_run_all)
    tests = [(name, fn) for name, fn in inspect.getmembers(this)
             if name.startswith('test_') and callable(fn)]
    passed = failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if _run_all() else 1)
