"""
Decision tree extraction and visualization for DAPTS strategies.

Given a solved DAPTS policy, extracts the full decision tree showing:
  - Which pool is chosen at each step
  - All possible outcomes (test results)
  - Cleared individuals at each node
  - Posterior beliefs after each observation

Supports text output and DOT format for Graphviz visualization.
"""

from augmented.core import mask_str, indices_from_mask, test_result, popcount
from augmented.bayesian import bayesian_update_single_test


def extract_tree(policy, p, u, n):
    """Extract the full decision tree from a DAPTS policy.

    Parameters
    ----------
    policy : DAPTS
        A solved DAPTS strategy.
    p : list[float]
        Prior latent-state probabilities.
    u : list[float]
        Individual utilities.
    n : int
        Population size.

    Returns
    -------
    dict
        Nested tree structure. Each node has:
          - 'step': test number (1-indexed)
          - 'pool': pool mask chosen
          - 'pool_str': human-readable pool
          - 'history': history leading to this node
          - 'cleared': cleared mask at this point
          - 'posteriors': current p values
          - 'children': dict mapping outcome r -> child node
          - 'utility': expected utility at terminal nodes
    """
    if getattr(policy, "cap", None) is not None:
        # A truncated policy keys histories on binned counts min(r, cap) and
        # its "r >= cap" branch aggregates several exact counts, so both the
        # raw-count branching below and bayesian_update_single_test (which
        # conditions on an exact count) would be wrong. Faithful rendering of
        # truncated trees needs range-conditioned posteriors; not yet done.
        raise NotImplementedError(
            "extract_tree does not support truncated (cap<G) policies; "
            f"got policy.cap={policy.cap}. Use cap=None (full counts)."
        )

    def _build(k, history, cleared_mask, current_p):
        if k > policy.B:
            # Terminal node
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': k,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }

        # Get the pool chosen at this step for this history
        try:
            pool = policy.choose(k, history)
        except KeyError:
            # History not reachable in the policy
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': k,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }

        pool_size = popcount(pool)
        children = {}

        if pool == 0:
            # Empty pool (wasted test) — only one outcome
            children[0] = _build(k + 1, history + ((0, 0),),
                                 cleared_mask, current_p)
        else:
            # Branch on all possible outcomes r = 0, 1, ..., |pool|
            for r in range(pool_size + 1):
                new_cleared = cleared_mask | pool if r == 0 else cleared_mask
                new_p = bayesian_update_single_test(current_p, pool, r, n)
                new_history = history + ((pool, r),)
                children[r] = _build(k + 1, new_history, new_cleared, new_p)

        return {
            'step': k,
            'terminal': False,
            'pool': pool,
            'pool_str': mask_str(pool, n),
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'children': children,
        }

    return _build(1, (), 0, list(p))


def print_tree(tree, n, indent=0, file=None):
    """Print a human-readable text representation of the decision tree.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    indent : int
        Current indentation level.
    file : file-like or None
        Output destination (default: stdout).
    """
    prefix = "  " * indent
    import sys
    out = file or sys.stdout

    if tree.get('terminal'):
        cleared = tree['cleared_str']
        post = [f"{pi:.3f}" for pi in tree['posteriors']]
        if 'pruned_note' in tree:
            out.write(f"{prefix}[PRUNED] cleared={cleared}"
                      f" {tree['pruned_note']}"
                      f" posteriors=[{', '.join(post)}]\n")
        else:
            utility = tree['utility']
            out.write(f"{prefix}[TERMINAL] cleared={cleared}"
                      f" utility={utility:.2f}"
                      f" posteriors=[{', '.join(post)}]\n")
        return

    step = tree['step']
    pool = tree['pool_str']
    cleared = tree['cleared_str']
    post = [f"{pi:.3f}" for pi in tree['posteriors']]

    out.write(f"{prefix}Step {step}: test pool {pool}"
              f" | cleared={cleared}"
              f" | p=[{', '.join(post)}]\n")

    for r in sorted(tree['children'].keys()):
        child = tree['children'][r]
        out.write(f"{prefix}  r={r}:\n")
        print_tree(child, n, indent + 2, file=out)


def tree_to_string(tree, n):
    """Return the tree as a string instead of printing."""
    import io
    buf = io.StringIO()
    print_tree(tree, n, file=buf)
    return buf.getvalue()


def prune_tree(tree, max_depth):
    """Return a deep copy of the tree pruned to max_depth levels.

    Nodes whose step exceeds max_depth are converted to terminal nodes
    with a note indicating how many subtrees were pruned.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    max_depth : int
        Maximum depth (step number) to keep.  Nodes at step > max_depth
        are collapsed into terminal placeholders.

    Returns
    -------
    dict
        A new tree (deep-copied) truncated at max_depth.
    """
    import copy

    def _count_subtrees(node):
        """Count total subtree nodes (including the node itself)."""
        if node.get('terminal'):
            return 1
        count = 1
        for child in node.get('children', {}).values():
            count += _count_subtrees(child)
        return count

    def _prune(node, depth):
        if node.get('terminal'):
            return copy.deepcopy(node)

        if depth >= max_depth:
            # Convert this non-terminal node into a pruned terminal node
            subtree_count = sum(
                _count_subtrees(ch)
                for ch in node.get('children', {}).values()
            )
            pruned = {
                'step': node['step'],
                'terminal': True,
                'history': node.get('history'),
                'cleared': node.get('cleared', 0),
                'cleared_str': node.get('cleared_str', ''),
                'posteriors': list(node.get('posteriors', [])),
                'pruned_note': f"[pruned: {subtree_count} subtrees]",
            }
            return pruned

        # Recurse into children
        new_node = {}
        for key, val in node.items():
            if key == 'children':
                new_node['children'] = {
                    r: _prune(child, depth + 1)
                    for r, child in val.items()
                }
            elif key == 'posteriors':
                new_node['posteriors'] = list(val)
            elif key == 'history':
                new_node['history'] = val
            else:
                new_node[key] = val
        return new_node

    return _prune(tree, 1)


def summarize_tree(tree, n):
    """Compute aggregate statistics over the decision tree.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.

    Returns
    -------
    dict
        Aggregate statistics with keys:
          - total_nodes: count of all nodes
          - terminal_nodes: count of terminal (leaf) nodes
          - max_depth: deepest step level in the tree
          - avg_branching: average branching factor of non-terminal nodes
          - avg_terminal_utility: average utility at terminal nodes
          - pools_used: set of unique pool strings tested
          - depth_distribution: dict mapping depth (step) -> node count
    """
    stats = {
        'total_nodes': 0,
        'terminal_nodes': 0,
        'max_depth': 0,
        'branching_factors': [],     # temporary; used to compute avg
        'terminal_utilities': [],    # temporary; used to compute avg
        'pools_used': set(),
        'depth_distribution': {},
    }

    def _walk(node):
        stats['total_nodes'] += 1

        depth = node.get('step', 0)
        if depth > stats['max_depth']:
            stats['max_depth'] = depth

        stats['depth_distribution'][depth] = (
            stats['depth_distribution'].get(depth, 0) + 1
        )

        if node.get('terminal'):
            stats['terminal_nodes'] += 1
            if 'utility' in node:
                stats['terminal_utilities'].append(node['utility'])
            return

        # Non-terminal node
        if 'pool_str' in node:
            stats['pools_used'].add(node['pool_str'])

        children = node.get('children', {})
        if children:
            stats['branching_factors'].append(len(children))
        for child in children.values():
            _walk(child)

    _walk(tree)

    # Compute averages
    bf = stats.pop('branching_factors')
    tu = stats.pop('terminal_utilities')
    stats['avg_branching'] = (sum(bf) / len(bf)) if bf else 0.0
    stats['avg_terminal_utility'] = (sum(tu) / len(tu)) if tu else 0.0

    return stats


def print_tree_summary(tree, n):
    """Print a human-readable one-paragraph summary of the decision tree.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    """
    s = summarize_tree(tree, n)
    pools = ', '.join(sorted(s['pools_used'])) if s['pools_used'] else 'none'
    depth_parts = ', '.join(
        f"depth {d}: {c}"
        for d, c in sorted(s['depth_distribution'].items())
    )
    print(
        f"Tree summary ({n} individuals): "
        f"{s['total_nodes']} total nodes, "
        f"{s['terminal_nodes']} terminal nodes, "
        f"maximum depth {s['max_depth']}, "
        f"average branching factor {s['avg_branching']:.2f}, "
        f"average terminal utility {s['avg_terminal_utility']:.4f}. "
        f"Pools tested: {pools}. "
        f"Depth distribution: {depth_parts}."
    )


def export_tree_dot(tree, n, title="DAPTS Decision Tree", max_depth=None):
    """Export the decision tree to DOT format for Graphviz.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    title : str
        Graph title.
    max_depth : int or None
        If provided, prune the tree to this depth before exporting.

    Returns
    -------
    str
        DOT-format string. Save to .dot file and render with:
        dot -Tpng tree.dot -o tree.png
    """
    if max_depth is not None:
        tree = prune_tree(tree, max_depth)
    lines = ['digraph DAPTS {']
    lines.append(f'  label="{title}";')
    lines.append('  labelloc="t";')
    lines.append('  node [shape=box, fontname="monospace", fontsize=10];')
    lines.append('  edge [fontname="monospace", fontsize=9];')
    lines.append('')

    node_id = [0]  # mutable counter

    def _add_node(tree_node, parent_id=None, edge_label=None):
        nid = node_id[0]
        node_id[0] += 1

        if tree_node.get('terminal'):
            cleared = tree_node['cleared_str']
            if 'pruned_note' in tree_node:
                note = tree_node['pruned_note']
                label = f"PRUNED\\ncleared={cleared}\\n{note}"
                lines.append(f'  n{nid} [label="{label}", '
                             f'style=filled, fillcolor="#fff3cd"];')
            else:
                utility = tree_node['utility']
                label = f"DONE\\ncleared={cleared}\\nutility={utility:.2f}"
                lines.append(f'  n{nid} [label="{label}", '
                             f'style=filled, fillcolor="#d4edda"];')
        else:
            step = tree_node['step']
            pool = tree_node['pool_str']
            cleared = tree_node['cleared_str']
            label = f"Step {step}\\ntest {pool}\\ncleared={cleared}"
            lines.append(f'  n{nid} [label="{label}", '
                         f'style=filled, fillcolor="#cce5ff"];')

        if parent_id is not None and edge_label is not None:
            lines.append(f'  n{parent_id} -> n{nid} '
                         f'[label=" r={edge_label} "];')

        if not tree_node.get('terminal') and 'children' in tree_node:
            for r in sorted(tree_node['children'].keys()):
                _add_node(tree_node['children'][r], nid, r)

    _add_node(tree)
    lines.append('}')
    return '\n'.join(lines)
