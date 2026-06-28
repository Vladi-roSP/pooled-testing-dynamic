# Beta Meta-parameter Implementation Plan

**Goal:** Add a standalone beta-reward greedy module that augments myopic pool selection with exact expected information gain and prints Sprint 2 benchmark tables.

**Architecture:** Create `augmented/infection_reward_greedy.py` as a self-contained companion to `augmented/greedy.py`. Keep the core beta-greedy policy exact over test outcomes, and restrict randomized scenario generation to the reporting helpers `run_vip_benchmark()` and `run_beta_sweep()`.

**Tech Stack:** Python, NumPy, existing `augmented.core`, `augmented.bayesian`, `augmented.greedy`, and `augmented.baselines`

---

### Task 1: Add the failing tests

**Files:**
- Create: `augmented/tests_infection_reward_greedy.py`

**Step 1: Write the failing test**

```python
def test_beta_zero_matches_standard_greedy_eu():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python augmented/tests_infection_reward_greedy.py`
Expected: FAIL because `augmented.infection_reward_greedy` does not exist yet.

### Task 2: Implement the beta-reward greedy module

**Files:**
- Create: `augmented/infection_reward_greedy.py`

**Step 1: Write minimal implementation**

- Implement `_compute_info_gain()` for `confirmed`, `entropy`, and `variance`
- Implement `_beta_best_pool()`
- Implement `greedy_myopic_beta_simulate()`
- Implement `greedy_myopic_beta_expected_utility()`
- Implement `run_vip_benchmark()` and `run_beta_sweep()`
- Add `if __name__ == "__main__":` entrypoint

**Step 2: Run tests to verify they pass**

Run: `python augmented/tests_infection_reward_greedy.py`
Expected: PASS

### Task 3: Verify the runnable module

**Files:**
- Verify: `augmented/infection_reward_greedy.py`

**Step 1: Run the module**

Run: `python augmented/infection_reward_greedy.py`
Expected: VIP benchmark table plus beta sweep summary.
