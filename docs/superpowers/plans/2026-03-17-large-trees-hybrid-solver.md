# Large Trees Exploration & Hybrid Solver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Jupyter notebook exploring large decision trees visually, plus a hybrid greedy→brute force solver and an infection-aware greedy meta-parameter.

**Architecture:** Three new files — `tree_visualizer.py` (Graphviz inline rendering), `hybrid_solver.py` (hybrid greedy→DP solver + infection-aware scoring), and `large_trees_exploration.ipynb` (6-block notebook). All code reuses existing modules in `augmented/`.

**Tech Stack:** Python, graphviz (Python package), IPython.display, matplotlib, numpy, existing augmented modules.

**Spec:** `docs/superpowers/specs/2026-03-17-large-trees-hybrid-solver-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `augmented/tree_visualizer.py` | Graphviz inline rendering for Jupyter: `render_tree`, `render_side_by_side`, `render_tree_series` |
| Create | `augmented/hybrid_solver.py` | Hybrid greedy→DP solver, branch value estimation, infection-aware scoring, info gain |
| Create | `augmented/notebooks/large_trees_exploration.ipynb` | 6-block exploratory notebook (Blocks 0-5) |
| Create | `augmented/tests_hybrid.py` | Tests for hybrid_solver.py |
| Create | `augmented/tests_visualizer.py` | Tests for tree_visualizer.py |

---

## Task 1: `tree_visualizer.py` — Core `render_tree`

**Files:**
- Create: `augmented/tree_visualizer.py`
- Create: `augmented/tests_visualizer.py`
- Read: `augmented/tree_extractor.py` (tree dict schema)

- [ ] **Step 1: Write failing test for `render_tree`**

```python
# augmented/tests_visualizer.py
"""Tests for tree_visualizer module."""

import graphviz
from augmented.solver import solve_optimal_dapts
from augmented.tree_extractor import extract_tree
from augmented.tree_visualizer import render_tree


def test_render_tree_returns_digraph():
    """render_tree returns a graphviz.Digraph for a simple n=2 tree."""
    p = [0.2, 0.3]
    u = [1.0, 1.0]
    val, policy = solve_optimal_dapts(p, u, B=1, G=2)
    tree = extract_tree(policy, p, u, n=2)
    result = render_tree(tree, n=2)
    assert isinstance(result, graphviz.Digraph)


def test_render_tree_has_nodes():
    """Rendered graph contains node definitions."""
    p = [0.2, 0.3]
    u = [1.0, 1.0]
    val, policy = solve_optimal_dapts(p, u, B=1, G=2)
    tree = extract_tree(policy, p, u, n=2)
    result = render_tree(tree, n=2)
    source = result.source
    assert 'label=' in source
    assert '->' in source  # has edges


def test_render_tree_with_group_colors():
    """Group colors apply without error."""
    p = [0.2, 0.3]
    u = [10.0, 1.0]
    val, policy = solve_optimal_dapts(p, u, B=1, G=2)
    tree = extract_tree(policy, p, u, n=2)
    colors = {0: 'red', 1: 'gray'}
    result = render_tree(tree, n=2, group_colors=colors)
    assert isinstance(result, graphviz.Digraph)


def test_render_tree_collapse():
    """Collapse threshold produces collapsed placeholder nodes."""
    p = [0.15, 0.15, 0.15, 0.15, 0.15]
    u = [1.0, 1.0, 1.0, 1.0, 1.0]
    val, policy = solve_optimal_dapts(p, u, B=4, G=5)
    tree = extract_tree(policy, p, u, n=5)
    result = render_tree(tree, n=5, collapse_threshold=3)
    assert '...' in result.source or 'nodes' in result.source


def test_render_tree_max_depth():
    """max_depth limits tree depth."""
    p = [0.15, 0.15, 0.15]
    u = [1.0, 1.0, 1.0]
    val, policy = solve_optimal_dapts(p, u, B=2, G=3)
    tree = extract_tree(policy, p, u, n=3)
    result = render_tree(tree, n=3, max_depth=1)
    assert isinstance(result, graphviz.Digraph)


def _run_all():
    """Run all tests."""
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
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_visualizer`
Expected: FAIL with "No module named 'augmented.tree_visualizer'" or ImportError

- [ ] **Step 3: Write `render_tree` implementation**

```python
# augmented/tree_visualizer.py
"""
Inline Graphviz visualization for DAPTS decision trees in Jupyter.

Wraps tree dicts from tree_extractor.extract_tree() into graphviz.Digraph
objects that render automatically in Jupyter via _repr_svg_().
"""

import graphviz
from augmented.core import indices_from_mask, mask_str
from augmented.tree_extractor import prune_tree


def _count_nodes(tree):
    """Count total nodes in a tree dict."""
    if tree.get('terminal'):
        return 1
    count = 1
    for child in tree.get('children', {}).values():
        count += _count_nodes(child)
    return count


def _utility_color(utility, max_utility):
    """Map utility to a green fill intensity. Higher = darker green."""
    if max_utility <= 0:
        return '#ffffff'
    ratio = min(1.0, max(0.0, utility / max_utility))
    # Interpolate from white (#ffffff) to green (#2ecc71)
    r = int(255 - ratio * (255 - 46))
    g = int(255 - ratio * (255 - 204))
    b = int(255 - ratio * (255 - 113))
    return f'#{r:02x}{g:02x}{b:02x}'


def _get_max_utility(tree):
    """Find max terminal utility in the tree."""
    if tree.get('terminal'):
        return tree.get('utility', 0)
    max_u = 0
    for child in tree.get('children', {}).values():
        max_u = max(max_u, _get_max_utility(child))
    return max_u


def render_tree(tree, n,
                group_colors=None,
                node_size_by='utility',
                show_posteriors=True,
                show_pool_labels=True,
                collapse_threshold=20,
                max_depth=None,
                title=None):
    """Render a decision tree inline in Jupyter.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    group_colors : dict or None
        Mapping agent_index -> color string for node border coloring.
    node_size_by : str
        'utility' for fill intensity by utility, 'fixed' for uniform.
    show_posteriors : bool
        Show posterior probabilities at each node.
    show_pool_labels : bool
        Show pool member labels on decision nodes.
    collapse_threshold : int
        Collapse subtrees with more than this many nodes.
    max_depth : int or None
        Prune tree to this depth before rendering.
    title : str or None
        Graph title.

    Returns
    -------
    graphviz.Digraph
    """
    if max_depth is not None:
        tree = prune_tree(tree, max_depth)

    max_utility = _get_max_utility(tree) if node_size_by == 'utility' else 1.0

    dot = graphviz.Digraph(format='svg')
    dot.attr(rankdir='TB')
    if title:
        dot.attr(label=title, labelloc='t', fontsize='14')
    dot.attr('node', fontname='monospace', fontsize='10')
    dot.attr('edge', fontname='monospace', fontsize='9')

    node_id = [0]

    def _pool_border_color(pool_mask):
        """Get border color based on group_colors of pool members."""
        if not group_colors:
            return '#333333'
        members = indices_from_mask(pool_mask, n)
        colors_in_pool = set(group_colors.get(i, '#333333') for i in members)
        if len(colors_in_pool) == 1:
            return colors_in_pool.pop()
        return '#333333'  # mixed group

    def _add_node(tree_node, parent_id=None, edge_label=None):
        nid = node_id[0]
        node_id[0] += 1
        nid_str = f'n{nid}'

        if tree_node.get('terminal'):
            cleared = tree_node.get('cleared_str', '{}')
            if 'pruned_note' in tree_node:
                note = tree_node['pruned_note']
                label = f'PRUNED\\ncleared={cleared}\\n{note}'
                dot.node(nid_str, label=label,
                         shape='box', style='filled,dashed',
                         fillcolor='#fff3cd', color='#856404')
            else:
                utility = tree_node.get('utility', 0)
                label = f'DONE\\ncleared={cleared}\\nutility={utility:.2f}'
                if show_posteriors and 'posteriors' in tree_node:
                    post = [f'{pi:.2f}' for pi in tree_node['posteriors']]
                    label += f'\\np=[{",".join(post)}]'
                fill = _utility_color(utility, max_utility) if node_size_by == 'utility' else '#d4edda'
                dot.node(nid_str, label=label,
                         shape='box', style='filled,rounded',
                         fillcolor=fill, color='#28a745')
        else:
            # Check if subtree should be collapsed
            subtree_size = _count_nodes(tree_node)
            if subtree_size > collapse_threshold and parent_id is not None:
                label = f'... ({subtree_size} nodes)'
                dot.node(nid_str, label=label,
                         shape='box', style='filled,dashed',
                         fillcolor='#f0f0f0', color='#999999')
                if parent_id is not None and edge_label is not None:
                    dot.edge(f'n{parent_id}', nid_str, label=f' r={edge_label} ')
                return

            step = tree_node['step']
            pool = tree_node.get('pool_str', '{}')
            cleared = tree_node.get('cleared_str', '{}')
            label = f'Step {step}'
            if show_pool_labels:
                label += f'\\ntest {pool}'
            label += f'\\ncleared={cleared}'
            if show_posteriors and 'posteriors' in tree_node:
                post = [f'{pi:.2f}' for pi in tree_node['posteriors']]
                label += f'\\np=[{",".join(post)}]'

            border = _pool_border_color(tree_node.get('pool', 0))
            dot.node(nid_str, label=label,
                     shape='box', style='filled',
                     fillcolor='#cce5ff', color=border, penwidth='2')

        if parent_id is not None and edge_label is not None:
            dot.edge(f'n{parent_id}', nid_str, label=f' r={edge_label} ')

        if not tree_node.get('terminal') and 'children' in tree_node:
            for r in sorted(tree_node['children'].keys()):
                _add_node(tree_node['children'][r], nid, r)

    _add_node(tree)
    return dot


def render_side_by_side(tree_a, tree_b, n,
                        title_a="A", title_b="B",
                        **kwargs):
    """Render two trees side by side using IPython HTML divs.

    Parameters
    ----------
    tree_a, tree_b : dict
        Trees from extract_tree().
    n : int
        Population size.
    title_a, title_b : str
        Titles for each tree.
    **kwargs
        Passed to render_tree.

    Returns
    -------
    IPython.display.HTML or graphviz.Digraph
        HTML with two inline SVGs if IPython available, else a combined Digraph.
    """
    dot_a = render_tree(tree_a, n, title=title_a, **kwargs)
    dot_b = render_tree(tree_b, n, title=title_b, **kwargs)
    try:
        from IPython.display import HTML
        svg_a = dot_a.pipe(format='svg').decode('utf-8')
        svg_b = dot_b.pipe(format='svg').decode('utf-8')
        html = (
            '<div style="display:flex;gap:20px;align-items:flex-start">'
            f'<div style="flex:1"><h4>{title_a}</h4>{svg_a}</div>'
            f'<div style="flex:1"><h4>{title_b}</h4>{svg_b}</div>'
            '</div>'
        )
        return HTML(html)
    except ImportError:
        # Fallback: combine as subgraphs
        combined = graphviz.Digraph()
        combined.attr(rankdir='TB')
        with combined.subgraph(name='cluster_a') as c:
            c.attr(label=title_a)
            c.subgraph(dot_a)
        with combined.subgraph(name='cluster_b') as c:
            c.attr(label=title_b)
            c.subgraph(dot_b)
        return combined


def render_tree_series(trees, n, titles, max_per_row=3, **kwargs):
    """Render multiple trees in a grid layout.

    Parameters
    ----------
    trees : list[dict]
        List of trees from extract_tree().
    n : int
        Population size.
    titles : list[str]
        Title for each tree.
    max_per_row : int
        Maximum trees per row (default 3).
    **kwargs
        Passed to render_tree.

    Returns
    -------
    IPython.display.HTML or list[graphviz.Digraph]
    """
    dots = [render_tree(t, n, title=ttl, **kwargs) for t, ttl in zip(trees, titles)]
    try:
        from IPython.display import HTML
        svgs = [d.pipe(format='svg').decode('utf-8') for d in dots]
        cells = []
        for i, (svg, ttl) in enumerate(zip(svgs, titles)):
            cells.append(
                f'<div style="flex:0 0 {100//max_per_row}%;padding:5px">'
                f'<h4>{ttl}</h4>{svg}</div>'
            )
        html = '<div style="display:flex;flex-wrap:wrap">' + ''.join(cells) + '</div>'
        return HTML(html)
    except ImportError:
        return dots
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_visualizer`
Expected: 5 passed, 0 failed

- [ ] **Step 5: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/tree_visualizer.py augmented/tests_visualizer.py
git commit -m "feat: add tree_visualizer module with inline Graphviz rendering"
```

---

## Task 2: `hybrid_solver.py` — Info gain & infection-aware scoring

**Files:**
- Create: `augmented/hybrid_solver.py`
- Create: `augmented/tests_hybrid.py`
- Read: `augmented/bayesian.py:15-33` (`_poisson_binomial_pmf`)
- Read: `augmented/greedy.py:29-60` (`_myopic_best_pool`)

- [ ] **Step 1: Write failing tests for entropy and info gain**

```python
# augmented/tests_hybrid.py
"""Tests for hybrid_solver module."""

import math
from augmented.hybrid_solver import (
    _safe_binary_entropy,
    expected_info_gain,
    infection_aware_score,
)
from augmented.core import mask_from_indices


def test_safe_binary_entropy_zero():
    """Entropy of deterministic variable is 0."""
    assert _safe_binary_entropy(0.0) == 0.0
    assert _safe_binary_entropy(1.0) == 0.0


def test_safe_binary_entropy_half():
    """Entropy of fair coin is 1 bit."""
    assert abs(_safe_binary_entropy(0.5) - 1.0) < 1e-10


def test_safe_binary_entropy_typical():
    """Entropy of p=0.2 is about 0.722 bits."""
    h = _safe_binary_entropy(0.2)
    expected = -0.2 * math.log2(0.2) - 0.8 * math.log2(0.8)
    assert abs(h - expected) < 1e-10


def test_info_gain_nonnegative():
    """Information gain is always non-negative."""
    p = [0.3, 0.2, 0.4]
    pool = mask_from_indices([0, 1])  # test persons 0 and 1
    ig = expected_info_gain(pool, p, n=3)
    assert ig >= -1e-10


def test_info_gain_deterministic_zero():
    """No info gain from testing people with p=0."""
    p = [0.0, 0.0, 0.5]
    pool = mask_from_indices([0, 1])
    ig = expected_info_gain(pool, p, n=3)
    assert abs(ig) < 1e-10


def test_infection_aware_score_alpha_one():
    """alpha=1.0 should give same ranking as standard myopic score."""
    p = [0.2, 0.3]
    u = [1.0, 1.0]
    pool = mask_from_indices([0, 1])
    score = infection_aware_score(pool, p, u, n=2, cleared_mask=0, alpha=1.0)
    # Standard myopic: P(r=0) * sum(u_i) = (0.8 * 0.7) * 2 = 1.12
    expected = 0.8 * 0.7 * 2.0
    assert abs(score - expected) < 1e-10


def test_infection_aware_score_alpha_zero():
    """alpha=0.0 should give pure info gain score."""
    p = [0.2, 0.3]
    u = [1.0, 1.0]
    pool = mask_from_indices([0, 1])
    score = infection_aware_score(pool, p, u, n=2, cleared_mask=0, alpha=0.0)
    ig = expected_info_gain(pool, p, n=2)
    assert abs(score - ig) < 1e-10


def _run_all():
    """Run all tests."""
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
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")


if __name__ == '__main__':
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: FAIL with ImportError

- [ ] **Step 3: Write entropy, info gain, and infection-aware scoring**

```python
# augmented/hybrid_solver.py
"""
Hybrid greedy→brute force solver for augmented pooled testing.

Combines greedy pool selection (fast, scalable) with exact DP
(optimal, expensive) to find near-optimal strategies for larger instances.

Also provides infection-aware scoring that balances clearing value
with information gain about infected individuals.
"""

import math
from augmented.core import (
    indices_from_mask, mask_str, popcount, mask_from_indices,
    all_pools_from_mask, compute_active_mask
)
from augmented.bayesian import (
    bayesian_update_single_test, _poisson_binomial_pmf
)
from augmented.greedy import _myopic_best_pool


# ---- Entropy / Information Gain ----

def _safe_binary_entropy(x):
    """Binary entropy H(x) in bits with 0*log(0) = 0 convention."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def expected_info_gain(pool_mask, p, n):
    """Expected reduction in posterior entropy from testing pool.

    Parameters
    ----------
    pool_mask : int
        Bitmask of the pool to test.
    p : list[float]
        Current infection probabilities.
    n : int
        Population size.

    Returns
    -------
    float
        Expected information gain (non-negative).
    """
    pool_idx = indices_from_mask(pool_mask, n)
    if not pool_idx:
        return 0.0

    # Entropy before testing (only for pool members)
    h_before = sum(_safe_binary_entropy(p[i]) for i in pool_idx)

    # For each outcome r, compute P(r) and posterior entropy
    pool_p = [p[i] for i in pool_idx]
    pmf = _poisson_binomial_pmf(pool_p)

    h_after = 0.0
    for r in range(len(pool_idx) + 1):
        if pmf[r] < 1e-15:
            continue
        # Compute posteriors given outcome r
        post_p = bayesian_update_single_test(list(p), pool_mask, r, n)
        h_r = sum(_safe_binary_entropy(post_p[i]) for i in pool_idx)
        h_after += pmf[r] * h_r

    return max(0.0, h_before - h_after)


def infection_aware_score(pool_mask, p, u, n, cleared_mask, alpha=0.5):
    """Modified pool scoring balancing clearing value with info gain.

    Score = alpha * P(r=0|t) * sum(u_i for i in t, uncleared)
          + (1-alpha) * expected_info_gain(t, p, n)

    Parameters
    ----------
    pool_mask : int
        Pool to evaluate.
    p : list[float]
        Current infection probabilities.
    u : list[float]
        Individual utilities.
    n : int
        Population size.
    cleared_mask : int
        Already-cleared individuals.
    alpha : float
        Tradeoff parameter in [0, 1].
        1.0 = pure clearing value, 0.0 = pure info gain.

    Returns
    -------
    float
        Combined score.
    """
    pool_idx = indices_from_mask(pool_mask, n)
    if not pool_idx:
        return 0.0

    # Clearing value: P(r=0) * sum(u_i for uncleared in pool)
    prob_clear = 1.0
    for i in pool_idx:
        prob_clear *= (1.0 - p[i])
    gain = sum(u[i] for i in pool_idx if not (cleared_mask >> i & 1))
    clearing_value = prob_clear * gain

    # Info gain
    ig = expected_info_gain(pool_mask, p, n)

    return alpha * clearing_value + (1.0 - alpha) * ig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: 7 passed, 0 failed

- [ ] **Step 5: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/hybrid_solver.py augmented/tests_hybrid.py
git commit -m "feat: add entropy, info gain, and infection-aware scoring to hybrid_solver"
```

---

## Task 3: `hybrid_solver.py` — Branch value estimation

**Files:**
- Modify: `augmented/hybrid_solver.py`
- Modify: `augmented/tests_hybrid.py`
- Read: `augmented/baselines.py` (`u_max`)
- Read: `augmented/greedy.py:90-118` (`greedy_myopic_expected_utility`)

- [ ] **Step 1: Write failing tests for `estimate_branch_value`**

Add to `augmented/tests_hybrid.py`:

```python
from augmented.hybrid_solver import estimate_branch_value
from augmented.baselines import u_max


def test_estimate_branch_value_bounds_order():
    """Lower bound <= upper bound."""
    p = [0.2, 0.3, 0.1]
    u = [1.0, 2.0, 1.5]
    lb, ub = estimate_branch_value(p, u, remaining_B=2, G=3, cleared_mask=0, n=3)
    assert lb <= ub + 1e-9


def test_estimate_branch_value_upper_is_umax():
    """Upper bound equals U_max of the active subpopulation."""
    p = [0.2, 0.3, 0.1]
    u = [1.0, 2.0, 1.5]
    _, ub = estimate_branch_value(p, u, remaining_B=2, G=3, cleared_mask=0, n=3)
    expected_ub = u_max(p, u)
    assert abs(ub - expected_ub) < 1e-10


def test_estimate_branch_value_with_cleared():
    """Cleared individuals contribute their full utility to both bounds."""
    p = [0.0, 0.3, 0.1]  # person 0 already known healthy
    u = [1.0, 2.0, 1.5]
    cleared_mask = 0b001  # person 0 cleared
    lb, ub = estimate_branch_value(p, u, remaining_B=1, G=2, cleared_mask=cleared_mask, n=3)
    # Both bounds should include u[0] = 1.0 from the cleared person
    assert lb >= 1.0 - 1e-10
    assert ub >= 1.0 - 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: New tests FAIL with ImportError for `estimate_branch_value`

- [ ] **Step 3: Implement `estimate_branch_value`**

Add to `augmented/hybrid_solver.py`:

```python
from augmented.baselines import u_max as _u_max
from augmented.greedy import greedy_myopic_expected_utility
# Note: compute_active_mask, indices_from_mask already imported at top of file


def estimate_branch_value(p_posterior, u, remaining_B, G, cleared_mask, n):
    """Estimate value achievable from a state without full DP.

    Parameters
    ----------
    p_posterior : list[float]
        Current posterior infection probabilities.
    u : list[float]
        Individual utilities.
    remaining_B : int
        Remaining test budget.
    G : int
        Max pool size.
    cleared_mask : int
        Already-cleared individuals bitmask.
    n : int
        Population size.

    Returns
    -------
    (lower_bound, upper_bound) : (float, float)
        lower_bound: greedy myopic expected utility on the subproblem
        upper_bound: U_max of active subpopulation + utility of already cleared
    """
    cleared_utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))

    # Upper bound: cleared utility + U_max of active (uncertain) population
    # Use compute_active_mask to properly exclude confirmed-infected individuals
    active_mask, _ = compute_active_mask(p_posterior, cleared_mask, n)
    active_indices = indices_from_mask(active_mask, n)
    active_p = [p_posterior[i] for i in active_indices]
    active_u = [u[i] for i in active_indices]

    upper_bound = cleared_utility + _u_max(active_p, active_u) if active_p else cleared_utility

    # Lower bound: greedy myopic on the subproblem
    if remaining_B > 0 and active_p:
        # We need to run greedy on a "virtual" subpoplem
        # Simplification: run greedy on the full state (it handles cleared_mask internally)
        lower_bound = greedy_myopic_expected_utility(p_posterior, u, remaining_B, G)
    else:
        lower_bound = cleared_utility

    # Numerical safety
    if lower_bound > upper_bound + 1e-9:
        # Anomaly — bounds inverted; treat as unreliable
        lower_bound = cleared_utility
        upper_bound = cleared_utility + sum(u)

    return lower_bound, upper_bound
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: 10 passed, 0 failed

- [ ] **Step 5: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/hybrid_solver.py augmented/tests_hybrid.py
git commit -m "feat: add branch value estimation to hybrid_solver"
```

---

## Task 4: `hybrid_solver.py` — Core hybrid greedy→brute force

**Files:**
- Modify: `augmented/hybrid_solver.py`
- Modify: `augmented/tests_hybrid.py`
- Read: `augmented/solver.py` (`solve_optimal_dapts`)
- Read: `augmented/tree_extractor.py:17-101` (`extract_tree`)

- [ ] **Step 1: Write failing tests for `hybrid_greedy_bruteforce`**

Add to `augmented/tests_hybrid.py`:

```python
from augmented.hybrid_solver import hybrid_greedy_bruteforce
from augmented.solver import solve_optimal_dapts
from augmented.greedy import greedy_myopic_expected_utility


def test_hybrid_k0_matches_dp():
    """K=0 greedy steps means full DP — should match optimal."""
    p = [0.2, 0.3, 0.15]
    u = [1.0, 2.0, 1.5]
    B, G = 2, 3
    tree, eu = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=0)
    dp_val, _ = solve_optimal_dapts(p, u, B, G)
    assert abs(eu - dp_val) < 1e-6, f"hybrid K=0 ({eu}) != DP ({dp_val})"


def test_hybrid_kB_matches_greedy():
    """K=B greedy steps means full greedy — should match greedy EU."""
    p = [0.2, 0.3, 0.15]
    u = [1.0, 2.0, 1.5]
    B, G = 2, 3
    tree, eu = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=B)
    greedy_val = greedy_myopic_expected_utility(p, u, B, G)
    assert abs(eu - greedy_val) < 1e-6, f"hybrid K=B ({eu}) != greedy ({greedy_val})"


def test_hybrid_returns_tree_dict():
    """Return value is a tree dict with expected keys."""
    p = [0.2, 0.3]
    u = [1.0, 1.0]
    tree, eu = hybrid_greedy_bruteforce(p, u, B=2, G=2, greedy_steps=1)
    assert isinstance(tree, dict)
    assert 'step' in tree
    assert 'terminal' in tree or 'children' in tree
    assert isinstance(eu, float)


def test_hybrid_monotonic():
    """Utility is non-decreasing as K decreases (more DP is better)."""
    p = [0.2, 0.3, 0.15, 0.25]
    u = [1.0, 2.0, 1.5, 1.0]
    B, G = 3, 3
    utilities = []
    for K in range(B + 1):
        _, eu = hybrid_greedy_bruteforce(p, u, B, G, greedy_steps=K)
        utilities.append(eu)
    # K=0 (full DP) should be >= K=1 >= ... >= K=B (full greedy)
    for i in range(len(utilities) - 1):
        assert utilities[i] >= utilities[i + 1] - 1e-6, \
            f"Monotonicity violated: K={i} ({utilities[i]}) < K={i+1} ({utilities[i+1]})"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: New tests FAIL with ImportError for `hybrid_greedy_bruteforce`

- [ ] **Step 3: Implement `hybrid_greedy_bruteforce`**

Add to `augmented/hybrid_solver.py`:

```python
from augmented.solver import solve_optimal_dapts
from augmented.tree_extractor import extract_tree


_MAX_N_DP = 14  # matches solver.py limit


def _infection_aware_best_pool(p, u, G, n, cleared_mask, alpha):
    """Pick pool maximizing infection_aware_score."""
    active_mask, _ = compute_active_mask(p, cleared_mask, n)
    if active_mask == 0:
        return 0
    pools = all_pools_from_mask(active_mask, G, include_empty=False)
    best_pool, best_score = 0, -1.0
    for pool in pools:
        score = infection_aware_score(pool, p, u, n, cleared_mask, alpha)
        if score > best_score:
            best_score = score
            best_pool = pool
    return best_pool


def hybrid_greedy_bruteforce(p, u, B, G, greedy_steps,
                              greedy_score_fn=None,
                              update_method='sequential'):
    """Hybrid solver: greedy for first K steps, then exact DP.

    Parameters
    ----------
    p : list[float]
        Prior infection probabilities.
    u : list[float]
        Individual utilities.
    B : int
        Total test budget.
    G : int
        Max pool size.
    greedy_steps : int
        Number of initial greedy steps (K). 0 = full DP, B = full greedy.
    greedy_score_fn : callable or None
        Custom pool scoring function(pool_mask, p, u, n, cleared_mask) -> float.
        If None, uses standard myopic scoring.
    update_method : str
        Bayesian update method for greedy phase: 'sequential' (default).

    Returns
    -------
    (tree_dict, expected_utility) : (dict, float)
        tree_dict matches tree_extractor.extract_tree schema.
    """
    n = len(p)

    def _select_pool(current_p, cleared_mask):
        if greedy_score_fn is not None:
            active_mask, _ = compute_active_mask(current_p, cleared_mask, n)
            if active_mask == 0:
                return 0
            pools = all_pools_from_mask(active_mask, G, include_empty=False)
            best_pool, best_score = 0, -1.0
            for pool in pools:
                score = greedy_score_fn(pool, current_p, u, n, cleared_mask)
                if score > best_score:
                    best_score = score
                    best_pool = pool
            return best_pool
        return _myopic_best_pool(current_p, u, G, n, cleared_mask)

    def _build_greedy_tree(step, current_p, cleared_mask, history, remaining_greedy):
        """Recursively build tree: greedy phase then DP phase."""

        # Phase 2: switch to DP
        if remaining_greedy == 0:
            remaining_budget = B - step + 1
            if remaining_budget <= 0:
                # No budget left — terminal
                utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
                return {
                    'step': step,
                    'terminal': True,
                    'history': history,
                    'cleared': cleared_mask,
                    'cleared_str': mask_str(cleared_mask, n),
                    'posteriors': list(current_p),
                    'utility': utility,
                }, utility

            # Count active agents for DP feasibility
            active_mask, _ = compute_active_mask(current_p, cleared_mask, n)
            n_active = popcount(active_mask)

            if n_active == 0 or remaining_budget == 0:
                utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
                return {
                    'step': step,
                    'terminal': True,
                    'history': history,
                    'cleared': cleared_mask,
                    'cleared_str': mask_str(cleared_mask, n),
                    'posteriors': list(current_p),
                    'utility': utility,
                }, utility

            if n_active <= _MAX_N_DP:
                # Build reduced subproblem with only active agents
                # (DP solver checks len(p) <= 14, so we must reduce)
                active_indices = indices_from_mask(active_mask, n)
                sub_p = [current_p[i] for i in active_indices]
                sub_u = [u[i] for i in active_indices]
                sub_n = len(active_indices)
                # Map: sub_index -> original_index
                idx_map = {si: oi for si, oi in enumerate(active_indices)}

                try:
                    dp_val_sub, dp_policy_sub = solve_optimal_dapts(
                        sub_p, sub_u, remaining_budget, min(G, sub_n)
                    )
                    dp_tree_sub = extract_tree(dp_policy_sub, sub_p, sub_u, sub_n)
                    # Remap sub-tree indices back to original population
                    _remap_tree_indices(dp_tree_sub, idx_map, n, current_p, cleared_mask, u)
                    # Adjust step numbers
                    _adjust_steps(dp_tree_sub, step - 1)
                    # Update history references
                    _update_history(dp_tree_sub, history)
                    # Add cleared utility to DP value
                    cleared_utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
                    return dp_tree_sub, dp_val_sub + cleared_utility
                except Exception:
                    pass  # fall through to greedy fallback

            # Fallback: continue with greedy for remaining budget
            return _build_greedy_tree(step, current_p, cleared_mask, history,
                                       remaining_budget)

        # Phase 1: greedy step
        pool = _select_pool(current_p, cleared_mask)

        if pool == 0:
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': step,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }, utility

        pool_idx = indices_from_mask(pool, n)
        pool_p = [current_p[i] for i in pool_idx]
        pmf = _poisson_binomial_pmf(pool_p)

        children = {}
        expected_utility = 0.0

        for r in range(len(pool_idx) + 1):
            if pmf[r] < 1e-15:
                continue
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_p = bayesian_update_single_test(current_p, pool, r, n)
            new_history = history + ((pool, r),)

            child_tree, child_eu = _build_greedy_tree(
                step + 1, new_p, new_cleared, new_history,
                remaining_greedy - 1
            )
            children[r] = child_tree
            expected_utility += pmf[r] * child_eu

        return {
            'step': step,
            'terminal': False,
            'pool': pool,
            'pool_str': mask_str(pool, n),
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'children': children,
        }, expected_utility

    tree, eu = _build_greedy_tree(1, list(p), 0, (), greedy_steps)
    return tree, eu


def _adjust_steps(tree, offset):
    """Add offset to all step numbers in a tree (in-place)."""
    tree['step'] += offset
    if not tree.get('terminal') and 'children' in tree:
        for child in tree['children'].values():
            _adjust_steps(child, offset)


def _update_history(tree, prefix_history):
    """Prepend prefix_history to all history tuples in tree (in-place)."""
    if 'history' in tree:
        tree['history'] = prefix_history + tree['history']
    if not tree.get('terminal') and 'children' in tree:
        for child in tree['children'].values():
            _update_history(child, prefix_history)


def _remap_tree_indices(tree, idx_map, n, full_p, cleared_mask, u):
    """Remap sub-problem tree indices back to original population (in-place).

    idx_map: dict mapping sub_index -> original_index
    Converts pool masks, cleared masks, posteriors, pool_str, cleared_str.
    """
    # Remap pool mask
    if 'pool' in tree and not tree.get('terminal'):
        sub_pool = tree['pool']
        new_pool = 0
        for si in indices_from_mask(sub_pool, len(idx_map)):
            new_pool |= 1 << idx_map[si]
        tree['pool'] = new_pool
        tree['pool_str'] = mask_str(new_pool, n)

    # Remap cleared mask: start with parent cleared, add sub-cleared
    if 'cleared' in tree:
        sub_cleared = tree['cleared']
        new_cleared = cleared_mask
        for si in indices_from_mask(sub_cleared, len(idx_map)):
            new_cleared |= 1 << idx_map[si]
        tree['cleared'] = new_cleared
        tree['cleared_str'] = mask_str(new_cleared, n)

    # Remap posteriors to full population
    if 'posteriors' in tree:
        sub_posteriors = tree['posteriors']
        full_posteriors = list(full_p)
        for si, oi in idx_map.items():
            if si < len(sub_posteriors):
                full_posteriors[oi] = sub_posteriors[si]
        tree['posteriors'] = full_posteriors

    # Remap utility at terminals
    if tree.get('terminal') and 'utility' in tree:
        tree['utility'] = sum(u[i] for i in indices_from_mask(tree['cleared'], n))

    # Remap history tuples
    if 'history' in tree:
        new_hist = []
        for pool_mask, r in tree['history']:
            new_pool = 0
            for si in indices_from_mask(pool_mask, len(idx_map)):
                if si in idx_map:
                    new_pool |= 1 << idx_map[si]
                else:
                    new_pool |= 1 << si  # already in original space
            new_hist.append((new_pool, r))
        tree['history'] = tuple(new_hist)

    # Recurse into children
    if not tree.get('terminal') and 'children' in tree:
        for child in tree['children'].values():
            _remap_tree_indices(child, idx_map, n, full_p, cleared_mask, u)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_hybrid`
Expected: 14 passed, 0 failed

- [ ] **Step 5: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/hybrid_solver.py augmented/tests_hybrid.py
git commit -m "feat: add hybrid greedy->brute force solver"
```

---

## Task 5: Notebook Block 0 — Building Intuition

**Files:**
- Create: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Create notebook with imports and Block 0.1 (simplest case)**

Create the notebook with cells:

**Cell 1** (markdown):
```markdown
# Large Trees Exploration & Hybrid Greedy→Brute Force

This notebook uses **visualization as the primary tool** to explore decision trees for augmented pooled testing. We progress from simple pedagogical examples to hybrid greedy→DP solvers.
```

**Cell 2** (code — imports):
```python
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML

sys.path.insert(0, '..')

from augmented.core import mask_from_indices, indices_from_mask, mask_str, popcount
from augmented.solver import solve_optimal_dapts
from augmented.classical_solver import solve_classical_dynamic
from augmented.greedy import (greedy_myopic_expected_utility,
                               greedy_myopic_counting_expected_utility)
from augmented.tree_extractor import extract_tree, summarize_tree, prune_tree
from augmented.tree_visualizer import render_tree, render_side_by_side, render_tree_series
from augmented.baselines import u_max, u_single
from augmented.hybrid_solver import (hybrid_greedy_bruteforce, estimate_branch_value,
                                      infection_aware_score, expected_info_gain,
                                      _infection_aware_best_pool)
from augmented.bayesian import bayesian_update_single_test, _poisson_binomial_pmf
from augmented.strategy import DAPTS
```

**Cell 3** (markdown):
```markdown
## Block 0: Building Intuition — Simple Pedagogical Trees

### 0.1 The Simplest Case — n=2, B=1, G=2
Two people, one test. Should we pool them together or test individually?
```

**Cell 4** (code):
```python
# Setup
p_01 = [0.2, 0.2]
u_01 = [1.0, 1.0]

# Optimal strategy
val_01, policy_01 = solve_optimal_dapts(p_01, u_01, B=1, G=2)
tree_01 = extract_tree(policy_01, p_01, u_01, n=2)

# Also solve with G=1 (individual only) for comparison
val_01_ind, policy_01_ind = solve_optimal_dapts(p_01, u_01, B=1, G=1)
tree_01_ind = extract_tree(policy_01_ind, p_01, u_01, n=2)

print(f"Pooled (G=2): EU = {val_01:.4f}")
print(f"Individual (G=1): EU = {val_01_ind:.4f}")
print(f"Advantage of pooling: {val_01 - val_01_ind:+.4f}")

display(render_side_by_side(tree_01_ind, tree_01, n=2,
                             title_a="Individual (G=1)", title_b="Pooled (G=2)",
                             show_posteriors=False))
```

**Cell 5** (markdown):
```markdown
### 0.2 The Value Effect — n=2, B=1, G=2, Asymmetric Utilities
Person A is very valuable (u=10), Person B is not (u=1). Same infection probability. How does the optimal strategy change?
```

**Cell 6** (code):
```python
p_02 = [0.1, 0.1]
u_02 = [10.0, 1.0]

val_02, policy_02 = solve_optimal_dapts(p_02, u_02, B=1, G=2)
tree_02 = extract_tree(policy_02, p_02, u_02, n=2)

print(f"Optimal EU = {val_02:.4f}")
display(render_tree(tree_02, n=2, group_colors={0: 'red', 1: 'gray'},
                     title="Optimal: u=[10, 1], p=[0.1, 0.1]"))

# Sweep u_A
u_a_values = np.arange(1, 21)
optimal_pools = []
eus = []
for ua in u_a_values:
    u_sweep = [float(ua), 1.0]
    val, pol = solve_optimal_dapts(p_02, u_sweep, B=1, G=2)
    tree = extract_tree(pol, p_02, u_sweep, n=2)
    pool = tree.get('pool_str', '{}')
    optimal_pools.append(pool)
    eus.append(val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(u_a_values, eus, 'b-o', markersize=4)
ax1.set_xlabel('u_A'); ax1.set_ylabel('Expected Utility')
ax1.set_title('EU vs u_A (u_B=1)')
ax1.grid(True, alpha=0.3)

# Show which pool is chosen
pool_labels = list(set(optimal_pools))
colors = plt.cm.Set2(np.linspace(0, 1, len(pool_labels)))
for i, ua in enumerate(u_a_values):
    c = colors[pool_labels.index(optimal_pools[i])]
    ax2.bar(ua, 1, color=c, edgecolor='none')
ax2.set_xlabel('u_A'); ax2.set_ylabel('Optimal pool choice')
ax2.set_title('Pool choice vs u_A')
# Legend
for j, lbl in enumerate(pool_labels):
    ax2.bar(0, 0, color=colors[j], label=lbl)
ax2.legend()
plt.tight_layout()
plt.show()
```

**Cell 7** (markdown):
```markdown
### 0.3 The Prevalence Effect — n=3, B=2, G=3
Three equal people, varying infection prevalence. Higher prevalence → smaller pools?
```

**Cell 8** (code):
```python
prevalences = [0.05, 0.2, 0.5]
trees_03 = []
titles_03 = []
for prev in prevalences:
    p_03 = [prev] * 3
    u_03 = [1.0] * 3
    val, pol = solve_optimal_dapts(p_03, u_03, B=2, G=3)
    tree = extract_tree(pol, p_03, u_03, n=3)
    trees_03.append(tree)
    titles_03.append(f"p={prev}")
    print(f"p={prev}: EU={val:.4f}")

display(render_tree_series(trees_03, n=3, titles=titles_03, show_posteriors=False))
```

**Cell 9** (markdown):
```markdown
### 0.4 Augmented vs Classical — n=3, B=2, G=3
Classical tests return binary (+/−), augmented tests return exact count. How does this change the decision tree?
```

**Cell 10** (code):
```python
p_04 = [0.2, 0.2, 0.2]
u_04 = [1.0, 1.0, 1.0]

# Augmented
val_aug, pol_aug = solve_optimal_dapts(p_04, u_04, B=2, G=3)
tree_aug = extract_tree(pol_aug, p_04, u_04, n=3)

# Classical — solve_classical_dynamic returns (value, None), no policy object
# We need to show the structural difference, so we'll annotate
val_cls, _ = solve_classical_dynamic(p_04, u_04, B=2, G=3)

print(f"Augmented EU = {val_aug:.4f}")
print(f"Classical EU  = {val_cls:.4f}")
print(f"Augmented advantage: {val_aug - val_cls:+.4f}")

# Show augmented tree — classical doesn't have a policy to extract
display(render_tree(tree_aug, n=3, title=f"Augmented (EU={val_aug:.4f})",
                     show_posteriors=True))

stats = summarize_tree(tree_aug, n=3)
print(f"\nAugmented tree: {stats['total_nodes']} nodes, "
      f"{stats['terminal_nodes']} terminals, "
      f"avg branching = {stats['avg_branching']:.2f}")
print("Note: Classical trees have binary branching (r=0 vs r>0), "
      "augmented trees branch on r=0,1,...,|pool|")
```

**Cell 11** (markdown):
```markdown
### 0.5 The Power of Budget — n=4, B ∈ {1,2,3,4}, G=4
More tests → more complex trees → higher utility. Watch the tree grow.
```

**Cell 12** (code):
```python
p_05 = [0.15] * 4
u_05 = [1.0] * 4
budgets = [1, 2, 3, 4]

trees_05 = []
titles_05 = []
vals_05 = []
for b in budgets:
    val, pol = solve_optimal_dapts(p_05, u_05, B=b, G=4)
    tree = extract_tree(pol, p_05, u_05, n=4)
    trees_05.append(tree)
    titles_05.append(f"B={b}, EU={val:.3f}")
    vals_05.append(val)

display(render_tree_series(trees_05, n=4, titles=titles_05,
                            show_posteriors=False, max_per_row=2))

# Utility vs B curve
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(budgets, vals_05, 'b-o', markersize=8, linewidth=2)
ax.axhline(y=u_max(p_05, u_05), color='r', linestyle='--', label='U_max (infinite budget)')
ax.set_xlabel('Budget (B)'); ax.set_ylabel('Expected Utility')
ax.set_title('EU vs Budget — n=4, G=4, p=0.15')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

**Cell 13** (markdown):
```markdown
### 0.6 "Why Not Test One by One?" — n=4, B=2, G ∈ {1,2,3,4}
Fixed budget, varying max pool size. The coverage-information tradeoff.
```

**Cell 14** (code):
```python
p_06 = [0.15] * 4
u_06 = [1.0] * 4
pool_sizes = [1, 2, 3, 4]

trees_06 = []
titles_06 = []
for g in pool_sizes:
    val, pol = solve_optimal_dapts(p_06, u_06, B=2, G=g)
    tree = extract_tree(pol, p_06, u_06, n=4)
    trees_06.append(tree)
    titles_06.append(f"G={g}, EU={val:.3f}")

display(render_tree_series(trees_06, n=4, titles=titles_06,
                            show_posteriors=False, max_per_row=2))
```

- [ ] **Step 2: Verify notebook runs**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && jupyter nbconvert --to notebook --execute augmented/notebooks/large_trees_exploration.ipynb --output /dev/null 2>&1 | tail -5`
Expected: No errors. (Or run interactively in Jupyter.)

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 0 — pedagogical tree examples"
```

---

## Task 6: Notebook Block 1 — Two Worlds

**Files:**
- Modify: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Add Block 1 cells**

**Cell 15** (markdown):
```markdown
---
## Block 1: "Two Worlds" — High-Value Group vs Common Group

4 VIPs (u=10, p=0.3) + 6 common people (u=1, p=0.1). B=5, G=4.
Does the optimal solver prioritize VIPs? Does greedy?
```

**Cell 16** (code):
```python
# Setup
n_b1 = 10
p_b1 = [0.3]*4 + [0.1]*6  # VIPs first, then common
u_b1 = [10.0]*4 + [1.0]*6
B_b1, G_b1 = 5, 4

vip_colors = {i: '#e74c3c' for i in range(4)}
vip_colors.update({i: '#95a5a6' for i in range(4, 10)})

# Optimal (DP)
print("Solving optimal DP (n=10, B=5, G=4)... ", end="", flush=True)
t0 = time.time()
val_opt, pol_opt = solve_optimal_dapts(p_b1, u_b1, B_b1, G_b1)
t_opt = time.time() - t0
print(f"done in {t_opt:.1f}s, EU={val_opt:.4f}")

tree_opt = extract_tree(pol_opt, p_b1, u_b1, n=n_b1)

# Greedy
val_greedy = greedy_myopic_expected_utility(p_b1, u_b1, B_b1, G_b1)
print(f"Greedy EU = {val_greedy:.4f}")
print(f"Gap: {val_opt - val_greedy:.4f} ({(val_opt - val_greedy)/val_opt*100:.1f}%)")

# Build greedy tree via hybrid with K=B
tree_greedy_b1, _ = hybrid_greedy_bruteforce(p_b1, u_b1, B_b1, G_b1, greedy_steps=B_b1)

# Side-by-side: Optimal vs Greedy
display(render_side_by_side(tree_opt, tree_greedy_b1, n=n_b1,
                             title_a=f"Optimal (EU={val_opt:.4f})",
                             title_b=f"Greedy (EU={val_greedy:.4f})",
                             group_colors=vip_colors,
                             show_posteriors=False, max_depth=3,
                             collapse_threshold=15))

# Metric table: utility breakdown by group
print("\nUtility breakdown by group (from optimal tree terminals):")
print(f"  U_max VIP: {sum(u_b1[i]*(1-p_b1[i]) for i in range(4)):.4f}")
print(f"  U_max Common: {sum(u_b1[i]*(1-p_b1[i]) for i in range(4,10)):.4f}")
```

**Cell 17** (code):
```python
# Sensitivity: vary VIP infection rate
p_vip_values = [0.1, 0.2, 0.3, 0.5]
results_b1 = []
for pv in p_vip_values:
    p_sens = [pv]*4 + [0.1]*6
    val, _ = solve_optimal_dapts(p_sens, u_b1, B_b1, G_b1)
    val_g = greedy_myopic_expected_utility(p_sens, u_b1, B_b1, G_b1)
    results_b1.append({'p_VIP': pv, 'Optimal': val, 'Greedy': val_g})
    print(f"p_VIP={pv}: Optimal={val:.4f}, Greedy={val_g:.4f}, Gap={val-val_g:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
x = [r['p_VIP'] for r in results_b1]
ax.plot(x, [r['Optimal'] for r in results_b1], 'b-o', label='Optimal', linewidth=2)
ax.plot(x, [r['Greedy'] for r in results_b1], 'r--s', label='Greedy', linewidth=2)
ax.axhline(y=u_max(p_b1, u_b1), color='gray', linestyle=':', label='U_max (original)')
ax.set_xlabel('VIP infection rate (p_VIP)')
ax.set_ylabel('Expected Utility')
ax.set_title('Two Worlds: EU vs VIP Infection Rate')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

- [ ] **Step 2: Verify notebook runs**

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 1 — Two Worlds (VIP vs common)"
```

---

## Task 7: Notebook Block 2 — Binary Search

**Files:**
- Modify: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Add Block 2 cells with binary search builder**

**Cell 18** (markdown):
```markdown
---
## Block 2: "Binary Search" — 8 Extremely Valuable People

n=8, u=[10]*8, p=0.15, B=10, G=8.
Can we use binary search to efficiently identify who's infected? Does the DP solver discover this structure naturally?
```

**Cell 19** (code):
```python
def build_binary_search_tree(agents, p, u, n, B, step=1, cleared_mask=0, history=()):
    """Recursively build a binary search decision tree.

    agents: list of agent indices to classify
    """
    if not agents or step > B:
        utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
        return {
            'step': step, 'terminal': True, 'history': history,
            'cleared': cleared_mask, 'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(p), 'utility': utility,
        }

    # Test all agents in the group
    pool = mask_from_indices(agents)
    pool_idx = agents
    pool_p = [p[i] for i in pool_idx]
    pmf = _poisson_binomial_pmf(pool_p)

    children = {}

    if len(agents) == 1:
        # Single person: test them individually
        for r in range(2):  # r=0 (healthy) or r=1 (infected)
            if pmf[r] < 1e-15:
                continue
            new_cleared = cleared_mask | pool if r == 0 else cleared_mask
            new_p = bayesian_update_single_test(list(p), pool, r, n)
            new_history = history + ((pool, r),)
            children[r] = build_binary_search_tree(
                [], new_p, u, n, B, step + 1, new_cleared, new_history
            )
    else:
        # Group test: if r=0, all clear; if r>0, split in half
        for r in range(len(agents) + 1):
            if pmf[r] < 1e-15:
                continue
            new_p = bayesian_update_single_test(list(p), pool, r, n)
            new_history = history + ((pool, r),)

            if r == 0:
                # All clear!
                new_cleared = cleared_mask | pool
                children[r] = build_binary_search_tree(
                    [], new_p, u, n, B, step + 1, new_cleared, new_history
                )
            else:
                # Split into two halves, test left half first
                mid = len(agents) // 2
                left = agents[:mid]
                right = agents[mid:]
                # Continue with left half (binary search)
                children[r] = build_binary_search_tree(
                    left, new_p, u, n, B, step + 1, cleared_mask, new_history
                )

    return {
        'step': step, 'terminal': False, 'pool': pool,
        'pool_str': mask_str(pool, n), 'history': history,
        'cleared': cleared_mask, 'cleared_str': mask_str(cleared_mask, n),
        'posteriors': list(p), 'children': children,
    }

# Setup
n_b2 = 8
p_b2 = [0.15] * n_b2
u_b2 = [10.0] * n_b2
B_b2, G_b2 = 10, 8

# Binary search tree
tree_bs = build_binary_search_tree(list(range(n_b2)), p_b2, u_b2, n_b2, B_b2)

# Optimal tree
print("Solving optimal DP (n=8, B=10, G=8)... ", end="", flush=True)
t0 = time.time()
val_dp, pol_dp = solve_optimal_dapts(p_b2, u_b2, B_b2, G_b2)
t_dp = time.time() - t0
print(f"done in {t_dp:.1f}s, EU={val_dp:.4f}")
tree_dp = extract_tree(pol_dp, p_b2, u_b2, n=n_b2)

# Greedy
val_greedy_b2 = greedy_myopic_expected_utility(p_b2, u_b2, B_b2, G_b2)
print(f"Greedy EU = {val_greedy_b2:.4f}")

# P(r=0) for first pool test (all 8)
p_all_clear = np.prod([1 - pi for pi in p_b2])
print(f"\nP(all 8 clear in one test) = {p_all_clear:.4f} ({p_all_clear*100:.1f}%)")
print(f"If clear: save 9 tests. Expected saving = {p_all_clear * 9:.2f} tests")
```

**Cell 20** (code):
```python
# Greedy tree
tree_greedy_b2, _ = hybrid_greedy_bruteforce(p_b2, u_b2, B_b2, G_b2, greedy_steps=B_b2)

# Side-by-side-by-side: Binary Search vs Optimal vs Greedy
display(render_tree_series(
    [tree_bs, tree_dp, tree_greedy_b2], n=n_b2,
    titles=["Binary Search", f"DP Optimal (EU={val_dp:.4f})",
            f"Greedy (EU={val_greedy_b2:.4f})"],
    show_posteriors=False, max_depth=3, collapse_threshold=10
))

# Efficiency metric: tests used vs fraction classified
# For each strategy, simulate on a few infection profiles and track
# how quickly individuals get classified
print("\nEfficiency: P(all clear in 1 test) = {:.1f}%".format(p_all_clear*100))
print("With binary search: 4 tests to fully classify in worst case (if 1 infected)")
print("With DP: solver may find more efficient classification paths")
```

- [ ] **Step 2: Verify notebook runs**

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 2 — Binary Search comparison"
```

---

## Task 8: Notebook Block 3 — Greedy vs Optimal

**Files:**
- Modify: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Add Block 3 cells**

**Cell 21** (markdown):
```markdown
---
## Block 3: Greedy vs Optimal on Larger Trees

Systematically compare greedy vs optimal across a grid of (n, B, G). Where does greedy diverge?
```

**Cell 22** (code):
```python
# Grid comparison (spec: n in {5,6,7,8}, B in {3,4,5}, G in {3,4})
grid_results = []
for n_g in [5, 6, 7, 8]:
    for B_g in [3, 4, 5]:
        for G_g in [3, 4]:
            if G_g > n_g:
                continue
            np.random.seed(42)
            p_g = list(np.random.uniform(0.05, 0.4, n_g))
            u_g = list(np.random.uniform(0.5, 3.0, n_g))

            val_dp, _ = solve_optimal_dapts(p_g, u_g, B_g, G_g)
            val_gr = greedy_myopic_expected_utility(p_g, u_g, B_g, G_g)
            gap = val_dp - val_gr
            gap_pct = (gap / val_dp * 100) if val_dp > 0 else 0

            grid_results.append({
                'n': n_g, 'B': B_g, 'G': G_g,
                'Optimal': val_dp, 'Greedy': val_gr,
                'Gap': gap, 'Gap%': gap_pct
            })
            print(f"n={n_g}, B={B_g}, G={G_g}: "
                  f"Opt={val_dp:.4f}, Greedy={val_gr:.4f}, Gap={gap_pct:.1f}%")

# Heatmap of gaps
fig, ax = plt.subplots(figsize=(8, 5))
configs = [f"n={r['n']},B={r['B']},G={r['G']}" for r in grid_results]
gaps = [r['Gap%'] for r in grid_results]
colors = plt.cm.YlOrRd([g / max(max(gaps), 1) for g in gaps])
bars = ax.barh(configs, gaps, color=colors)
ax.set_xlabel('Optimality Gap (%)')
ax.set_title('Greedy vs Optimal: Gap Across Configurations')
for bar, g in zip(bars, gaps):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{g:.1f}%', va='center', fontsize=9)
plt.tight_layout(); plt.show()
```

**Cell 23** (code):
```python
# Detailed tree comparison for worst-gap case
worst = max(grid_results, key=lambda r: r['Gap%'])
print(f"Worst gap: n={worst['n']}, B={worst['B']}, G={worst['G']} ({worst['Gap%']:.1f}%)")

np.random.seed(42)
p_worst = list(np.random.uniform(0.05, 0.4, worst['n']))
u_worst = list(np.random.uniform(0.5, 3.0, worst['n']))

val_dp_w, pol_dp_w = solve_optimal_dapts(p_worst, u_worst, worst['B'], worst['G'])
tree_dp_w = extract_tree(pol_dp_w, p_worst, u_worst, n=worst['n'])

# Build greedy tree via hybrid with K=B
tree_gr_w, _ = hybrid_greedy_bruteforce(p_worst, u_worst, worst['B'], worst['G'],
                                         greedy_steps=worst['B'])

display(render_side_by_side(tree_dp_w, tree_gr_w, n=worst['n'],
                             title_a=f"Optimal (EU={worst['Optimal']:.4f})",
                             title_b=f"Greedy (EU={worst['Greedy']:.4f})",
                             show_posteriors=True, max_depth=3))

# Greedy variants: compare myopic-sequential vs counting vs gibbs
print("\nGreedy variants on worst-gap instance:")
val_seq = greedy_myopic_expected_utility(p_worst, u_worst, worst['B'], worst['G'])
val_cnt = greedy_myopic_counting_expected_utility(p_worst, u_worst, worst['B'], worst['G'])
print(f"  Myopic-sequential: EU={val_seq:.4f}")
print(f"  Myopic-counting:   EU={val_cnt:.4f}")
print(f"  Optimal:           EU={worst['Optimal']:.4f}")
print(f"  Counting vs Sequential gap: {val_cnt - val_seq:+.4f}")
```

- [ ] **Step 2: Verify notebook runs**

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 3 — Greedy vs Optimal grid comparison"
```

---

## Task 9: Notebook Block 4 — Hybrid Greedy→Brute Force

**Files:**
- Modify: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Add Block 4 cells**

**Cell 24** (markdown):
```markdown
---
## Block 4: Hybrid Greedy→Brute Force

The key question: can we get near-optimal utility with just a few DP steps at the end?

**Sweep K** (number of greedy steps): K=0 is full DP, K=B is full greedy.
```

**Cell 25** (code):
```python
# Primary scenario
n_b4, B_b4, G_b4 = 8, 6, 4
np.random.seed(123)
p_b4 = list(np.random.uniform(0.1, 0.35, n_b4))
u_b4 = list(np.random.uniform(1.0, 5.0, n_b4))

print(f"Scenario: n={n_b4}, B={B_b4}, G={G_b4}")
print(f"p = [{', '.join(f'{pi:.3f}' for pi in p_b4)}]")
print(f"u = [{', '.join(f'{ui:.3f}' for ui in u_b4)}]")

# Sweep K
results_b4 = []
for K in range(B_b4 + 1):
    t0 = time.time()
    tree, eu = hybrid_greedy_bruteforce(p_b4, u_b4, B_b4, G_b4, greedy_steps=K)
    elapsed = time.time() - t0
    stats = summarize_tree(tree, n_b4)
    results_b4.append({
        'K': K, 'EU': eu, 'time': elapsed,
        'nodes': stats['total_nodes'], 'terminals': stats['terminal_nodes']
    })
    label = "full DP" if K == 0 else ("full greedy" if K == B_b4 else f"K={K}")
    print(f"K={K} ({label}): EU={eu:.4f}, time={elapsed:.2f}s, nodes={stats['total_nodes']}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

Ks = [r['K'] for r in results_b4]
ax = axes[0]
ax.plot(Ks, [r['EU'] for r in results_b4], 'b-o', linewidth=2)
ax.set_xlabel('Greedy steps (K)'); ax.set_ylabel('Expected Utility')
ax.set_title('EU vs Greedy Steps')
ax.axhline(y=results_b4[0]['EU'], color='g', linestyle='--', alpha=0.5, label='Full DP')
ax.axhline(y=results_b4[-1]['EU'], color='r', linestyle='--', alpha=0.5, label='Full Greedy')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(Ks, [r['time'] for r in results_b4], 'r-s', linewidth=2)
ax.set_xlabel('Greedy steps (K)'); ax.set_ylabel('Time (seconds, log)')
ax.set_title('Computation Time vs K')
ax.grid(True, alpha=0.3)

ax = axes[2]
# Gap to optimal
optimal_eu = results_b4[0]['EU']
gaps = [(optimal_eu - r['EU']) / optimal_eu * 100 if optimal_eu > 0 else 0
        for r in results_b4]
ax.bar(Ks, gaps, color='orange', alpha=0.7)
ax.set_xlabel('Greedy steps (K)'); ax.set_ylabel('Gap to Optimal (%)')
ax.set_title('Optimality Gap vs K')
ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
```

**Cell 26** (code):
```python
# Branch value estimation after K=2 greedy steps
K_est = 2
tree_k2, eu_k2 = hybrid_greedy_bruteforce(p_b4, u_b4, B_b4, G_b4, greedy_steps=K_est)

# Collect leaf states after K_est greedy steps
def collect_leaves_at_depth(tree, depth):
    """Collect nodes at a specific depth."""
    if tree.get('terminal') or tree['step'] > depth:
        return [tree]
    if tree['step'] == depth and not tree.get('terminal'):
        # This node is at the target depth; collect its children
        leaves = []
        for child in tree.get('children', {}).values():
            leaves.append(child)
        return leaves
    leaves = []
    for child in tree.get('children', {}).values():
        leaves.extend(collect_leaves_at_depth(child, depth))
    return leaves

# Get states after K_est steps
leaves = collect_leaves_at_depth(tree_k2, K_est + 1)
print(f"After K={K_est} greedy steps: {len(leaves)} leaf states")

# Estimate branch values
for i, leaf in enumerate(leaves[:8]):  # show up to 8
    posteriors = leaf.get('posteriors', p_b4)
    cleared = leaf.get('cleared', 0)
    remaining_B = B_b4 - K_est
    lb, ub = estimate_branch_value(posteriors, u_b4, remaining_B, G_b4, cleared, n_b4)
    tightness = (ub - lb) / ub * 100 if ub > 0 else 0
    print(f"  Leaf {i}: cleared={mask_str(cleared, n_b4)}, "
          f"LB={lb:.3f}, UB={ub:.3f}, gap={tightness:.1f}%")
```

**Cell 27** (code):
```python
# Visualize hybrid tree with phase boundary
# K=2: first 2 levels are greedy (blue), rest is DP (green)
sweet_spot = min(range(1, B_b4), key=lambda k:
    abs(results_b4[k]['EU'] - 0.99 * results_b4[0]['EU']))
print(f"Sweet spot: K={sweet_spot} gets within 1% of optimal")

tree_sweet, eu_sweet = hybrid_greedy_bruteforce(p_b4, u_b4, B_b4, G_b4,
                                                  greedy_steps=sweet_spot)
display(render_tree(tree_sweet, n=n_b4,
                     title=f"Hybrid K={sweet_spot} (EU={eu_sweet:.4f})",
                     show_posteriors=False, max_depth=4, collapse_threshold=15))

# Comparison table: Hybrid(K=2) vs full greedy vs full DP
print("\n" + "=" * 65)
print(f"{'Strategy':<25} {'EU':>8} {'Time(s)':>8} {'Nodes':>8}")
print("-" * 65)
for label, K in [("Full DP (K=0)", 0), ("Hybrid (K=2)", 2), ("Full Greedy (K=B)", B_b4)]:
    r = results_b4[K]
    print(f"{label:<25} {r['EU']:>8.4f} {r['time']:>8.2f} {r['nodes']:>8}")
print("=" * 65)
```

**Cell 28** (markdown):
```markdown
### Budget Allocation Analogy
"50 tests, 48 greedy, 2 remaining" — can we estimate remaining value without full DP?
```

**Cell 29** (code):
```python
# Conceptual: n=12, B=10, greedy for 8 steps, estimate remaining 2
n_ba, B_ba, G_ba = 12, 10, 4
np.random.seed(99)
p_ba = list(np.random.uniform(0.05, 0.25, n_ba))
u_ba = list(np.random.uniform(1.0, 5.0, n_ba))

K_greedy = 8
tree_ba, eu_ba = hybrid_greedy_bruteforce(p_ba, u_ba, B_ba, G_ba, greedy_steps=K_greedy)
print(f"n={n_ba}, B={B_ba}, Greedy K={K_greedy}: EU={eu_ba:.4f}")

# Full greedy baseline
_, eu_full_greedy = hybrid_greedy_bruteforce(p_ba, u_ba, B_ba, G_ba, greedy_steps=B_ba)
print(f"Full greedy: EU={eu_full_greedy:.4f}")
print(f"Hybrid improvement: {eu_ba - eu_full_greedy:+.4f}")

# Estimate branch values at depth K_greedy
leaves_ba = collect_leaves_at_depth(tree_ba, K_greedy + 1)
print(f"\nAfter K={K_greedy} steps: {len(leaves_ba)} leaf states")
for i, leaf in enumerate(leaves_ba[:5]):
    posteriors = leaf.get('posteriors', p_ba)
    cleared = leaf.get('cleared', 0)
    lb, ub = estimate_branch_value(posteriors, u_ba, B_ba - K_greedy, G_ba, cleared, n_ba)
    print(f"  Leaf {i}: LB={lb:.3f}, UB={ub:.3f}, gap={ub-lb:.3f}")
```

- [ ] **Step 2: Verify notebook runs**

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 4 — Hybrid greedy→brute force with K-sweep"
```

---

## Task 10: Notebook Block 5 — Infection-Aware Greedy

**Files:**
- Modify: `augmented/notebooks/large_trees_exploration.ipynb`

- [ ] **Step 1: Add Block 5 cells**

**Cell 28** (markdown):
```markdown
---
## Block 5: Infection-Aware Greedy (Meta-parameter α)

Standard greedy only values clearing healthy people. Can we do better by also valuing *information about who's infected*?

α=1.0: pure clearing value (standard greedy)
α=0.0: pure information gain
```

**Cell 29** (code):
```python
# Setup
n_b5, B_b5, G_b5 = 6, 4, 3
p_b5 = [0.3] * n_b5
u_b5 = [1.0] * n_b5

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
trees_b5 = []
eus_b5 = []

for alpha in alphas:
    score_fn = lambda pool, p, u, n, cm, a=alpha: infection_aware_score(
        pool, p, u, n, cm, alpha=a
    )
    tree, eu = hybrid_greedy_bruteforce(p_b5, u_b5, B_b5, G_b5,
                                         greedy_steps=B_b5,
                                         greedy_score_fn=score_fn)
    trees_b5.append(tree)
    eus_b5.append(eu)
    print(f"α={alpha:.2f}: EU={eu:.4f}")

# Compare against optimal
val_opt_b5, _ = solve_optimal_dapts(p_b5, u_b5, B_b5, G_b5)
print(f"\nOptimal EU = {val_opt_b5:.4f}")

# Plot EU vs alpha
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(alphas, eus_b5, 'b-o', linewidth=2, markersize=8, label='Infection-aware greedy')
ax.axhline(y=val_opt_b5, color='g', linestyle='--', label=f'Optimal ({val_opt_b5:.4f})')
std_greedy = eus_b5[-1]  # alpha=1.0
ax.axhline(y=std_greedy, color='r', linestyle=':', label=f'Standard greedy ({std_greedy:.4f})')
ax.set_xlabel('α (1=clearing, 0=info gain)')
ax.set_ylabel('Expected Utility')
ax.set_title('Infection-Aware Greedy: EU vs α')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

**Cell 30** (code):
```python
# Show trees for selected alphas
best_alpha_idx = max(range(len(alphas)), key=lambda i: eus_b5[i])
print(f"Best α = {alphas[best_alpha_idx]:.2f} (EU = {eus_b5[best_alpha_idx]:.4f})")

display(render_tree_series(
    [trees_b5[0], trees_b5[best_alpha_idx], trees_b5[-1]], n=n_b5,
    titles=[f"α=0 (info, EU={eus_b5[0]:.3f})",
            f"α={alphas[best_alpha_idx]} (best, EU={eus_b5[best_alpha_idx]:.3f})",
            f"α=1 (standard, EU={eus_b5[-1]:.3f})"],
    show_posteriors=False, max_depth=3
))
```

**Cell 31** (code):
```python
# Info gain visualization for a specific state
print("Information gain per pool (initial state):")
from augmented.core import all_pools_from_mask, compute_active_mask

active_mask, _ = compute_active_mask(p_b5, 0, n_b5)
pools = all_pools_from_mask(active_mask, G_b5, include_empty=False)

pool_info = []
for pool in pools[:20]:  # top 20
    ig = expected_info_gain(pool, p_b5, n_b5)
    pool_idx = indices_from_mask(pool, n_b5)
    prob_clear = np.prod([1 - p_b5[i] for i in pool_idx])
    clearing = prob_clear * sum(u_b5[i] for i in pool_idx)
    pool_info.append({
        'pool': mask_str(pool, n_b5), 'info_gain': ig,
        'clearing': clearing, 'combined_05': 0.5 * clearing + 0.5 * ig
    })

pool_info.sort(key=lambda x: x['combined_05'], reverse=True)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(pool_info))
w = 0.3
ax.bar([i - w for i in x], [pi['clearing'] for pi in pool_info], w, label='Clearing value', color='#3498db')
ax.bar(list(x), [pi['info_gain'] for pi in pool_info], w, label='Info gain', color='#e74c3c')
ax.bar([i + w for i in x], [pi['combined_05'] for pi in pool_info], w, label='Combined (α=0.5)', color='#2ecc71')
ax.set_xticks(list(x))
ax.set_xticklabels([pi['pool'] for pi in pool_info], rotation=45, fontsize=8)
ax.set_ylabel('Score'); ax.set_title('Pool Scoring: Clearing vs Info Gain')
ax.legend(); plt.tight_layout(); plt.show()
```

**Cell 32** (code):
```python
# Connection to hybrid: does a smarter Phase 1 improve hybrid?
print("Hybrid comparison: standard vs infection-aware greedy as Phase 1")
print("=" * 60)

K_hybrid = 2
# Standard greedy Phase 1
tree_std, eu_std = hybrid_greedy_bruteforce(p_b5, u_b5, B_b5, G_b5, greedy_steps=K_hybrid)
print(f"Standard greedy (K={K_hybrid}): EU = {eu_std:.4f}")

# Infection-aware Phase 1 (best alpha)
best_alpha = alphas[best_alpha_idx]
score_fn_best = lambda pool, p, u, n, cm: infection_aware_score(
    pool, p, u, n, cm, alpha=best_alpha
)
tree_aware, eu_aware = hybrid_greedy_bruteforce(
    p_b5, u_b5, B_b5, G_b5, greedy_steps=K_hybrid,
    greedy_score_fn=score_fn_best
)
print(f"Infection-aware (α={best_alpha}, K={K_hybrid}): EU = {eu_aware:.4f}")
print(f"Improvement: {eu_aware - eu_std:+.4f}")
print(f"Optimal: {val_opt_b5:.4f}")

display(render_side_by_side(tree_std, tree_aware, n=n_b5,
                             title_a=f"Standard Hybrid (EU={eu_std:.4f})",
                             title_b=f"Infection-aware Hybrid (EU={eu_aware:.4f})",
                             show_posteriors=False, max_depth=3))
```

**Cell 33** (markdown):
```markdown
### "Hunting Infecteds" — When Information About Infecteds Matters
A scenario where standard greedy ignores a clearly-infected individual, but infection-aware greedy tests them to gain information that helps clear others.
```

**Cell 34** (code):
```python
# Scenario: person 0 almost certainly infected (p=0.95), persons 1-4 uncertain
# Standard greedy avoids person 0 (high p = low P(r=0) = low clearing value)
# Infection-aware greedy includes person 0 for information gain
p_hunt = [0.95, 0.3, 0.3, 0.3, 0.3]
u_hunt = [1.0, 2.0, 2.0, 2.0, 2.0]
n_hunt, B_hunt, G_hunt = 5, 3, 3

# Standard greedy
tree_std_hunt, eu_std_hunt = hybrid_greedy_bruteforce(
    p_hunt, u_hunt, B_hunt, G_hunt, greedy_steps=B_hunt)
print(f"Standard greedy: EU={eu_std_hunt:.4f}")
print(f"  First pool: {tree_std_hunt.get('pool_str', 'N/A')}")

# Infection-aware (alpha=0.3)
score_hunt = lambda pool, p, u, n, cm: infection_aware_score(pool, p, u, n, cm, alpha=0.3)
tree_aware_hunt, eu_aware_hunt = hybrid_greedy_bruteforce(
    p_hunt, u_hunt, B_hunt, G_hunt, greedy_steps=B_hunt, greedy_score_fn=score_hunt)
print(f"Infection-aware (α=0.3): EU={eu_aware_hunt:.4f}")
print(f"  First pool: {tree_aware_hunt.get('pool_str', 'N/A')}")

# Optimal
val_opt_hunt, pol_opt_hunt = solve_optimal_dapts(p_hunt, u_hunt, B_hunt, G_hunt)
print(f"Optimal: EU={val_opt_hunt:.4f}")

display(render_tree_series(
    [tree_std_hunt, tree_aware_hunt], n=n_hunt,
    titles=[f"Standard (EU={eu_std_hunt:.3f})", f"Info-aware (EU={eu_aware_hunt:.3f})"],
    show_posteriors=True, max_depth=3
))
print("\nKey insight: including the near-certain infected (person 0) in the pool")
print("gives augmented count info that helps classify the uncertain persons.")
```

- [ ] **Step 2: Verify notebook runs**

- [ ] **Step 3: Commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add augmented/notebooks/large_trees_exploration.ipynb
git commit -m "feat: add Block 5 — Infection-aware greedy meta-parameter"
```

---

## Task 11: Final verification and cleanup

**Files:**
- All created files

- [ ] **Step 1: Run all tests**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests_visualizer && python -m augmented.tests_hybrid && python -m augmented.tests`
Expected: All pass.

- [ ] **Step 2: Run the existing test suite to check for regressions**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && python -m augmented.tests`
Expected: All existing tests pass.

- [ ] **Step 3: Execute notebook end-to-end**

Run: `cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic && jupyter nbconvert --to notebook --execute augmented/notebooks/large_trees_exploration.ipynb --ExecutePreprocessor.timeout=600`
Expected: Executes without errors. Output cells populated.

- [ ] **Step 4: Final commit**

```bash
cd /Users/hectorbecerrilvillamil/Desktop/PooledTesting.nosync/pooled-testing-dynamic
git add -A
git commit -m "feat: complete large trees exploration notebook with hybrid solver"
```
