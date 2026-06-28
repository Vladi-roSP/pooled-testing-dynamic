# Sprint 3 Experiments Implementation Plan

**Goal:** Add a standalone Sprint 3 experiment runner that writes incremental CSV results for the main scale study plus VIP, utility-modulation, and large-G sweeps.

**Architecture:** Create `augmented/sprint3_experiments.py` as a companion to `augmented/overnight_experiments.py`, reusing its incremental CSV and GC patterns while adding task-specific instance generators and measurement helpers. Keep timing, warning, and CSV-writing logic centralized so all experiment families share the same behavior.

**Tech Stack:** Python, csv, datetime, gc, time, existing `augmented` baselines, greedy solvers, pool solvers, and beta-greedy module

---

### Task 1: Add the failing smoke tests

**Files:**
- Create: `augmented/tests_sprint3_experiments.py`

**Step 1: Write the failing test**

```python
def test_timed_call_soft_timeout_keeps_value():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python augmented/tests_sprint3_experiments.py`
Expected: FAIL because `augmented.sprint3_experiments` does not exist yet.

### Task 2: Implement the Sprint 3 runner

**Files:**
- Create: `augmented/sprint3_experiments.py`

**Step 1: Write minimal implementation**

- Add timestamped CSV helpers and timed-call helpers
- Add main experiment configs and random-instance generators
- Add `run_main_experiments()`
- Add `run_vip_experiments()`
- Add `run_utility_modulation()`
- Add `run_large_G()`
- Add argparse entrypoint with `--configs`, `--n-instances`, and `--quick`

**Step 2: Run tests to verify they pass**

Run: `python augmented/tests_sprint3_experiments.py`
Expected: PASS

### Task 3: Verify the standalone script

**Files:**
- Verify: `augmented/sprint3_experiments.py`

**Step 1: Run the quick mode**

Run: `python augmented/sprint3_experiments.py --quick`
Expected: completes without errors and writes CSV files to `results/`.
