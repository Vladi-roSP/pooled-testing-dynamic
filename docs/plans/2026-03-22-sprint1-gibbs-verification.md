# Sprint 1 Gibbs Verification Implementation Plan

**Goal:** Add Phase 3 Sprint 1 Gibbs verification coverage and a standalone convergence analysis script in `augmented/`.

**Architecture:** Extend `augmented/tests.py` with deliberate slow verification tests that compare Gibbs posterior estimates and Gibbs-based greedy expected utility against exact counting on small instances. Add a separate `augmented/gibbs_analysis.py` script that prints convergence tables for fixed instances without changing any existing production logic.

**Tech Stack:** Python, NumPy, existing `augmented` Bayesian and greedy utilities

---

### Task 1: Add the new Gibbs verification tests

**Files:**
- Modify: `augmented/tests.py`

**Step 1: Write the failing tests**

- Add `test_gibbs_systematic_exact_comparison()` with the exact configs, random seeds, random instance generation, realistic histories via `greedy_myopic_simulate`, printed summary metrics, and the `fraction_passed >= 0.95` assertion.
- Add `test_gibbs_greedy_vs_counting_eu()` with the exact configs, seeds, per-instance printed output, and the `fraction_passed >= 0.90` assertion.

**Step 2: Run the targeted tests to verify they fail or expose any missing imports**

Run: `python -m pytest augmented/tests.py -k "systematic_exact_comparison or greedy_vs_counting_eu" -v`

Expected: failure if imports/helpers are missing before the test file is fully updated.

**Step 3: Write the minimal implementation**

- Add any needed local helpers/imports in `augmented/tests.py`.
- Keep all logic inside the test module so existing production code remains untouched.

**Step 4: Run the targeted tests to verify they pass**

Run: `python -m pytest augmented/tests.py -k "systematic_exact_comparison or greedy_vs_counting_eu" -v`

Expected: PASS

### Task 2: Add the standalone Gibbs convergence analysis script

**Files:**
- Create: `augmented/gibbs_analysis.py`

**Step 1: Write the script implementation**

- Make it runnable as `python augmented/gibbs_analysis.py`.
- Compute exact counting posteriors for the fixed `n=8` and `n=6` instances.
- For each iteration count and seed, compute the max absolute posterior error and print the mean/std table.

**Step 2: Run the script to verify output**

Run: `python augmented/gibbs_analysis.py`

Expected: two printed convergence tables, one for `n=8` and one for `n=6`.

### Task 3: Final verification

**Files:**
- Verify: `augmented/tests.py`
- Verify: `augmented/gibbs_analysis.py`

**Step 1: Run the targeted verification commands**

Run: `python augmented/tests.py`
Run: `python augmented/gibbs_analysis.py`

Expected: all tests pass, and the analysis script prints convergence summaries.
