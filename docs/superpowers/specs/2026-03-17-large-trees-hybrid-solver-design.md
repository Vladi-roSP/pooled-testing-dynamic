# Large Trees Exploration & Hybrid Greedy→Brute Force Solver

**Date**: 2026-03-17
**Status**: Reviewed
**Scope**: New notebook (`large_trees_exploration.ipynb`), new module (`hybrid_solver.py`), new visualization wrapper (`tree_visualizer.py`)

---

## 1. Goal

Create an interactive Jupyter notebook that uses **visualization as the primary tool** to:

1. Build intuition with simple, pedagogical decision trees
2. Explore how optimal and greedy solvers handle scenarios with heterogeneous groups (high-value vs low-value individuals)
3. Compare binary search strategies against DP-optimal trees for highly valuable populations
4. Implement and evaluate a **hybrid greedy→brute force solver** that uses greedy for early decisions and exact DP for the remaining budget
5. Explore a **meta-parameter** that modifies greedy scoring to also value identifying infected individuals (not just clearing healthy ones)

All notebook content in **English**.

---

## 2. Architecture

### 2.1 New Files

```
augmented/
├── hybrid_solver.py          # Hybrid greedy→brute force solver
├── tree_visualizer.py        # Graphviz-based inline Jupyter visualization
└── notebooks/
    └── large_trees_exploration.ipynb
```

### 2.2 Dependencies on Existing Code

| Existing Module | What We Use |
|---|---|
| `core.py` | `mask_from_indices`, `indices_from_mask`, `mask_str`, `popcount`, `test_result`, `all_pools`, `compute_active_mask` |
| `solver.py` | `solve_optimal_dapts` — full DP solver for exact comparison and for brute force phase of hybrid |
| `classical_solver.py` | `solve_classical_dynamic` — classical solver for augmented vs classical comparison |
| `greedy.py` | `_myopic_best_pool` (private, intentional coupling — see note), `greedy_myopic_expected_utility` — greedy phase of hybrid |
| `bayesian.py` | `bayesian_update_single_test`, `bayesian_update_by_counting`, `_poisson_binomial_pmf` |
| `strategy.py` | `DAPTS` — policy representation |
| `tree_extractor.py` | `extract_tree`, `summarize_tree`, `export_tree_dot`, `prune_tree` |
| `baselines.py` | `u_max`, `u_single` — bounds (note: lowercase function names) |
| `expected_utility.py` | `exact_expected_utility` — validate hybrid utility |
| `comparison.py` | `compare_all` — benchmark against all 8 strategies + U_max upper bound |

> **Note on `_myopic_best_pool`**: This is a private function in `greedy.py`. We use it directly in `hybrid_solver.py` as intentional internal coupling. If `greedy.py` is refactored, `hybrid_solver.py` must be updated accordingly.

---

## 3. Module: `tree_visualizer.py`

Wraps `tree_extractor.py` output into inline Graphviz rendering for Jupyter.

### 3.1 Public API

```python
def render_tree(tree, n,
                group_colors=None,      # dict: agent_index → color string
                node_size_by='utility',  # 'utility' | 'fixed'
                show_posteriors=True,
                show_pool_labels=True,
                collapse_threshold=20,   # collapse subtrees with > N nodes
                max_depth=None,
                title=None) -> graphviz.Digraph:
    """Render a decision tree inline in Jupyter."""

def render_side_by_side(tree_a, tree_b, n,
                        title_a="A", title_b="B",
                        **kwargs) -> graphviz.Digraph:
    """Render two trees in a single Graphviz figure (subgraphs) for comparison."""

def render_tree_series(trees, n, titles, **kwargs) -> graphviz.Digraph:
    """Render a horizontal series of trees (e.g., for parameter sweeps)."""
```

### 3.2 Visual Encoding

| Element | Encoding |
|---|---|
| Decision node (test) | Rectangle, labeled with pool members `{0,2,3}` |
| Terminal node (leaf) | Rounded box, labeled with cleared set + utility |
| Edge | Labeled with outcome `r=0`, `r=1`, etc. |
| Group membership | Node border color from `group_colors` dict |
| Posterior annotation | Small text below node: `p=[0.1, 0.0, 0.3, ...]` (rounded) |
| Collapsed subtree | Dashed node: `"... (47 nodes)"` |
| Utility magnitude | Node fill intensity proportional to cumulative utility |

### 3.3 Implementation Notes

- Use `graphviz` Python package (not shell `dot` command)
- Return `graphviz.Digraph` object — Jupyter renders it automatically via `_repr_svg_()`
- For side-by-side: prefer `IPython.display.HTML` with two `<div style="display:inline-block">` elements, each containing a separate SVG render. This avoids Graphviz's known layout issues with subgraph clusters of different-sized trees. Fallback to subgraph clusters if HTML rendering is unavailable.
- Collapse logic: DFS count subtree size; if > threshold, replace with summary node
- For `render_tree_series` with many trees (>3): use a grid layout (max 3 per row) via HTML divs to prevent very wide output

---

## 4. Module: `hybrid_solver.py`

### 4.1 Core Algorithm

```python
def hybrid_greedy_bruteforce(p, u, B, G, greedy_steps,
                              greedy_score_fn=None,
                              update_method='sequential'):
    """
    Phase 1 (greedy): Run greedy pool selection for `greedy_steps` steps.
                      At each step, select pool, branch on all possible outcomes,
                      update posteriors. Builds a tree dict (same schema as
                      tree_extractor.extract_tree) during this phase.
    Phase 2 (brute force): For each leaf state after Phase 1, invoke the
                           DP solver on the subproblem:
                           - remaining budget: B - greedy_steps
                           - active population: agents not yet cleared/confirmed
                           - posteriors: updated from Phase 1 observations
                           Then extract the DP subtree and graft it onto the
                           corresponding leaf of the greedy-phase tree.

    Returns: (tree_dict, expected_utility)
        tree_dict: nested dict matching tree_extractor schema:
            {step, pool, pool_str, history, cleared, cleared_str,
             posteriors, children: {r: child_node}, terminal}
        expected_utility: float, weighted expected utility over all leaf states
    """
```

**Data structure**: The hybrid solver builds a tree dict directly (not a DAPTS object). During Phase 1, greedy decisions are recorded as tree nodes. At each Phase 1 leaf, the DP solver produces a DAPTS sub-policy, which is converted to a subtree via `extract_tree` and grafted onto the leaf. This avoids the complexity of stitching DAPTS objects from mixed sources.

**Key implementation detail**: After `greedy_steps` greedy decisions, the population at each leaf may be small enough (n_active ≤ 14) for the DP solver. If n_active > 14, fall back to continued greedy for that branch (also recorded as tree nodes).

**Timeout/fallback**: The DP phase for each leaf has a configurable timeout (default 60s). If exceeded, that branch falls back to greedy continuation. The n=10 stretch scenario in Block 4 is explicitly optional/best-effort.

### 4.2 Branch Value Estimation

```python
def estimate_branch_value(p_posterior, u, remaining_B, G, cleared_mask, n):
    """
    Estimate the value achievable from a state without running full DP.

    Returns: (lower_bound, upper_bound)
        lower_bound: greedy myopic expected utility on the subproblem
        upper_bound: U_max of the active subpopulation (sum of u_i for all
                     uncertain agents weighted by P(healthy))
    """
```

This enables a **branch-and-bound** style decision: if `lower_bound ≈ upper_bound`, skip the expensive DP solve for that branch.

**Numerical safety**: Assert `lower_bound <= upper_bound + 1e-9`. If violated (due to floating-point issues in Poisson-Binomial PMF edge cases), fall back to always running DP for that branch.

### 4.3 Modified Greedy Scoring (Meta-parameter)

```python
def infection_aware_score(pool_mask, p, u, n, cleared_mask, alpha=0.5):
    """
    Modified pool scoring that balances clearing value with infection detection.

    Score(t) = alpha * P(r=0|t) * sum(u_i for i in t)           # clearing value
             + (1 - alpha) * expected_info_gain(t, p, n)         # information gain

    where expected_info_gain measures expected reduction in posterior entropy
    across all possible outcomes r=0,1,...,|t|.
    """
```

The `alpha` parameter controls the tradeoff:
- `alpha=1.0`: pure greedy (maximize clearing, ignore info about infecteds)
- `alpha=0.0`: pure info gain (maximize posterior certainty, ignore clearing)
- `alpha=0.5`: balanced

### 4.4 Entropy / Information Gain

```python
def _safe_binary_entropy(x):
    """Binary entropy with 0*log(0) = 0 convention."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)

def expected_info_gain(pool_mask, p, n):
    """
    Expected reduction in posterior entropy from testing pool t.

    H_before = sum(_safe_binary_entropy(p_i)) for i in pool
    H_after  = sum over outcomes r: P(r|t) * H(posteriors given r)
    info_gain = H_before - H_after

    Uses _safe_binary_entropy to handle p_i = 0 or p_i = 1 (common after
    Bayesian updates) without producing NaN.
    """
```

Uses the Poisson-Binomial PMF (already in `bayesian.py`) to compute outcome probabilities and posterior updates.

---

## 5. Notebook: `large_trees_exploration.ipynb`

### Block 0: Building Intuition — Simple Pedagogical Trees

All subsections compare trees visually with `render_tree` and `render_side_by_side`.

#### 0.1 The Simplest Case — n=2, B=1, G=2
- **Setup**: 2 people, 1 test, pool size up to 2. Equal p=0.2, u=1.
- **Show**: Two possible strategies as trees: (a) test person 0 alone, (b) pool {0,1}
- **Annotate**: Expected utility for each. Which is better and why?
- **Lesson**: When pooling beats individual testing.

#### 0.2 The Value Effect — n=2, B=1, G=2, Asymmetric Utilities
- **Setup**: Person A: u=10, p=0.1. Person B: u=1, p=0.1.
- **Show**: Optimal tree. Does it test A alone, B alone, or pool?
- **Vary**: Sweep u_A from 1→20 with a plot of optimal action vs u_A.
- **Lesson**: Utility asymmetry biases toward protecting high-value individuals.

#### 0.3 The Prevalence Effect — n=3, B=2, G=3
- **Setup**: 3 equal people (u=1), sweep p ∈ {0.05, 0.2, 0.5}.
- **Show**: 3 optimal trees side by side (one per prevalence level).
- **Lesson**: Higher prevalence → smaller pools, more fragmentation.

#### 0.4 Augmented vs Classical — n=3, B=2, G=3
- **Setup**: Same scenario, compare classical (binary +/−) vs augmented (exact count) trees.
- **Show**: Side-by-side trees. Highlight the extra branches augmented has (r=0,1,2 vs r=0,1).
- **Lesson**: Count information changes tree structure and improves utility.

#### 0.5 The Power of Budget — n=4, B ∈ {1,2,3,4}, G=4
- **Setup**: 4 equal people (u=1, p=0.15), vary B.
- **Show**: Series of 4 trees growing in complexity.
- **Metric**: Utility vs B curve alongside.
- **Lesson**: Each additional test opens branches and increases expected utility.

#### 0.6 "Why Not Test One by One?" — n=4, B=2, G ∈ {1,2,3,4}
- **Setup**: Fix B=2, vary max pool size.
- **Show**: 4 trees (G=1 individual → G=4 full pool).
- **Lesson**: The coverage-information tradeoff of pooling.

---

### Block 1: "Two Worlds" — High-Value Group vs Common Group

#### Setup
- n=10 total: 4 VIP (u=10, p=0.3) + 6 common (u=1, p=0.1)
- B=5, G=4
- `group_colors = {0:'red', 1:'red', 2:'red', 3:'red', 4:'gray', ...}`

#### Analyses
1. **Optimal tree** (DP, n=10 may require greedy-counting as DP maxes at n=14): Visualize with group colors. Does the solver prioritize VIPs?
2. **Greedy tree** (myopic sequential): Same visualization. Does greedy also prioritize VIPs?
3. **Side-by-side**: Optimal vs greedy. Where do they diverge?
4. **Metric table**: Expected utility breakdown by group (VIP cleared utility vs common cleared utility).
5. **Sensitivity**: Vary the VIP infection rate p_VIP ∈ {0.1, 0.2, 0.3, 0.5}. How does the optimal strategy adapt?

---

### Block 2: "Binary Search" — 8 Extremely Valuable People

#### Setup
- n=8, u=[10]*8, p=0.15 (uniform), B=10, G=8

#### Analyses
1. **Manual binary search tree**: Programmatically construct a DAPTS via a recursive `build_binary_search_dapts(n, B, G)` helper:
   - The helper recursively assigns pool = left_half at each step, and for each outcome branches: r=0 → clear that half, r>0 → subdivide further.
   - Step 1: Pool all 8. If r=0 → all clear. If r>0 → split {0-3} vs {4-7}.
   - Step 2-3: Test halves. If r=0 → half is clear. If r>0 → split again.
   - Steps 4+: Continue until individuals identified or budget exhausted.
   - The helper is defined in the notebook (not a new module) since it's a one-off pedagogical tool.
   - Visualize this tree.

2. **Optimal tree** (DP): Solve and visualize. Compare structure to binary search.

3. **Greedy tree**: Visualize. Does greedy approximate binary search?

4. **Side-by-side-by-side**: Binary search vs optimal vs greedy.

5. **Efficiency metric**: For each strategy, plot "number of tests used" vs "fraction of population classified" (a step function showing how quickly each strategy resolves individuals).

6. **"Is it worth it?" question**: After the first pool test (all 8), if r=0 we save 9 remaining tests. What's P(r=0) = (1-0.15)^8 ≈ 0.27? Annotate this on the tree.

---

### Block 3: Greedy vs Optimal on Larger Trees

#### Setup
- Grid: n ∈ {5, 6, 7, 8}, B ∈ {3, 4, 5}, G ∈ {3, 4}
- Random instances + designed instances

#### Analyses
1. **Divergence map**: For each (n, B, G), compute greedy vs optimal expected utility gap. Heatmap.
2. **Tree diff**: For selected high-divergence cases, show trees side-by-side with **divergent branches highlighted in red**.
3. **Greedy variants**: Compare myopic-sequential vs myopic-counting vs myopic-gibbs trees. Where does the Bayesian update method change the greedy decision?
4. **Annotation**: On each divergent branch, annotate: "Greedy chose {0,1,2}, Optimal chose {0,3,4}. Utility difference: +0.3"

---

### Block 4: Hybrid Greedy→Brute Force

#### Setup
- n=8, B=6, G=4 (primary scenario)
- Also: n=10, B=8, G=5 (stretch scenario, may need adaptive fallback)

#### Analyses
1. **Sweep K** (greedy steps): K = 0 (full DP), 1, 2, ..., 6 (full greedy).
   - Plot: Expected utility vs K.
   - Plot: Computation time vs K.
   - Identify the "sweet spot" where utility is near-optimal but computation is tractable.

2. **Branch value estimation**: After K=2 greedy steps, for each leaf state:
   - Compute: `lower_bound` (greedy continuation), `upper_bound` (U_max of active subpop), `exact_value` (DP solve)
   - Table + bar chart showing bounds tightness.
   - **Key question**: Can we skip DP for branches where `lower ≈ upper`?

3. **Hybrid tree visualization**: Show the full hybrid tree with a visual boundary between greedy phase (blue nodes) and DP phase (green nodes).

4. **Comparison table**: Hybrid(K=2) vs full greedy vs full DP — utility, runtime, tree size.

5. **Budget allocation analogy**: "50 tests, 48 greedy, 2 remaining" scenario.
   - n=12, B=50 (conceptual — too large for DP, but can show greedy phase + estimate remaining value)
   - Show greedy tree for 48 steps (pruned), then estimate branch values for remaining 2 tests.

---

### Block 5: Infection-Aware Greedy (Meta-parameter)

#### Setup
- n=6, B=4, G=3, p=[0.3]*6, u=[1]*6

#### Analyses
1. **Alpha sweep**: α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
   - For each α, run modified greedy and extract tree.
   - Show 5 trees in a row (or collapsed views).
   - Plot: Expected utility vs α.

2. **Information gain visualization**: For a specific state, show the info gain of each candidate pool as a bar chart. Compare against the standard clearing-value score.

3. **Connection to hybrid**: Use best α as the greedy phase scoring in the hybrid solver.
   - Compare: hybrid(standard greedy, K=2) vs hybrid(infection-aware greedy α=0.3, K=2)
   - Does a smarter Phase 1 reduce the gap to optimal?

4. **"Hunting infecteds" example**: Scenario where standard greedy ignores a clearly-infected individual (high p), but infection-aware greedy tests them to gain information that helps clear others.

---

## 6. Implementation Constraints

- **DP solver limit**: n ≤ 14 (from `solver.py:_MAX_N`). Scenarios with n > 14 use greedy + estimation only.
- **Visualization**: All trees rendered inline via `graphviz` Python package. No external `dot` calls.
- **Notebook language**: All text, comments, and markdown cells in **English**.
- **Code reuse**: Import all existing modules. No reimplementation of core algorithms.
- **Performance**: For the sweep analyses (Block 4 K-sweep, Block 5 α-sweep), cache results in a notebook-level dict keyed by `(tuple(p), tuple(u), B, G, K)` or `(tuple(p), tuple(u), B, G, alpha)`. Note: `lru_cache` won't work since `p` and `u` are lists (unhashable).

---

## 7. Success Criteria

1. Block 0 trees render correctly and tell a clear pedagogical story
2. Block 1-2 show visually distinct strategies for heterogeneous vs homogeneous populations
3. Block 3 highlights where greedy diverges from optimal with clear annotations
4. Block 4 demonstrates a working hybrid solver with measurable utility-vs-cost tradeoff
5. Block 5 shows the infection-aware meta-parameter changes greedy behavior in meaningful ways
6. All visualizations are inline, readable, and color-coded by group where applicable
7. `hybrid_solver.py` passes basic correctness tests: (a) utility matches DP for K=0, (b) matches greedy for K=B, (c) utility is monotonically non-decreasing as K decreases from B toward 0 (more DP always helps or stays the same)
