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
