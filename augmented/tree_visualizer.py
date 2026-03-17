"""
Graphviz inline rendering for DAPTS decision trees in Jupyter notebooks.

Provides functions to render decision trees extracted by tree_extractor as
interactive Graphviz diagrams.  The returned objects render inline in Jupyter
via their ``_repr_svg_()`` method.
"""

import graphviz
from augmented.tree_extractor import prune_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_nodes(tree):
    """DFS count of all nodes in a tree dict."""
    if tree.get('terminal'):
        return 1
    count = 1
    for child in tree.get('children', {}).values():
        count += _count_nodes(child)
    return count


def _get_max_utility(tree):
    """Find the maximum terminal utility in the tree."""
    if tree.get('terminal'):
        return tree.get('utility', 0.0)
    best = 0.0
    for child in tree.get('children', {}).values():
        best = max(best, _get_max_utility(child))
    return best


def _utility_color(utility, max_utility):
    """Map a utility value to a green intensity hex color.

    Higher utility -> more saturated green.  Returns a hex string like
    ``#d4edda`` (light green) to ``#28a745`` (deep green).
    """
    if max_utility <= 0:
        return '#d4edda'
    ratio = min(utility / max_utility, 1.0)
    # Interpolate from light green (212, 237, 218) to deep green (40, 167, 69)
    r = int(212 + (40 - 212) * ratio)
    g = int(237 + (167 - 237) * ratio)
    b = int(218 + (69 - 218) * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


def _posteriors_label(posteriors):
    """Format posteriors as a compact string."""
    parts = [f'{pi:.3f}' for pi in posteriors]
    return f'p=[{", ".join(parts)}]'


# ---------------------------------------------------------------------------
# Main rendering function
# ---------------------------------------------------------------------------

def render_tree(tree, n, group_colors=None, node_size_by='utility',
                show_posteriors=True, show_pool_labels=True,
                collapse_threshold=20, max_depth=None, title=None):
    """Render a DAPTS decision tree as a Graphviz Digraph.

    Parameters
    ----------
    tree : dict
        Tree from ``extract_tree()``.
    n : int
        Population size.
    group_colors : dict or None
        Maps agent index -> color string for node borders.  When a pool
        contains agents with different colors, the first color is used.
    node_size_by : str
        Currently unused; reserved for future sizing strategies.
    show_posteriors : bool
        If True, show posterior probabilities on each node.
    show_pool_labels : bool
        If True, show the pool label on decision nodes.
    collapse_threshold : int
        If a subtree has more than this many nodes it is collapsed into a
        single placeholder node.
    max_depth : int or None
        If given, prune the tree to this depth before rendering.
    title : str or None
        Graph title.

    Returns
    -------
    graphviz.Digraph
        Object that renders inline in Jupyter via ``_repr_svg_()``.
    """
    if max_depth is not None:
        tree = prune_tree(tree, max_depth)

    max_util = _get_max_utility(tree)

    dot = graphviz.Digraph(format='svg')
    dot.attr(rankdir='TB')
    if title:
        dot.attr(label=title, labelloc='t', fontsize='14')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    _node_counter = [0]

    def _new_id():
        nid = _node_counter[0]
        _node_counter[0] += 1
        return f'n{nid}'

    def _border_color(pool_mask):
        """Determine border color from group_colors and pool members."""
        if not group_colors:
            return 'black'
        for i in range(n):
            if pool_mask >> i & 1:
                if i in group_colors:
                    return group_colors[i]
        return 'black'

    def _add_subtree(node, parent_id=None, edge_label=None):
        nid = _new_id()

        # --- Collapsed subtree ---
        if not node.get('terminal') and collapse_threshold is not None:
            node_count = _count_nodes(node)
            if node_count > collapse_threshold:
                label = f'... ({node_count} nodes)'
                dot.node(nid, label=label, shape='box',
                         style='dashed', color='gray', fontcolor='gray')
                if parent_id is not None:
                    dot.edge(parent_id, nid,
                             label=f' r={edge_label} ' if edge_label is not None else '')
                return

        # --- Terminal node ---
        if node.get('terminal'):
            cleared = node.get('cleared_str', '{}')
            lines = []

            if 'pruned_note' in node:
                lines.append('PRUNED')
                lines.append(f'cleared={cleared}')
                lines.append(node['pruned_note'])
                if show_posteriors and node.get('posteriors'):
                    lines.append(_posteriors_label(node['posteriors']))
                label = '\n'.join(lines)
                dot.node(nid, label=label, shape='box',
                         style='dashed,filled,rounded', fillcolor='#fff3cd',
                         color='orange')
            else:
                utility = node.get('utility', 0.0)
                lines.append('DONE')
                lines.append(f'cleared={cleared}')
                lines.append(f'utility={utility:.2f}')
                if show_posteriors and node.get('posteriors'):
                    lines.append(_posteriors_label(node['posteriors']))
                label = '\n'.join(lines)
                fill = _utility_color(utility, max_util)
                dot.node(nid, label=label, shape='box',
                         style='filled,rounded', fillcolor=fill,
                         color='black')

            if parent_id is not None:
                dot.edge(parent_id, nid,
                         label=f' r={edge_label} ' if edge_label is not None else '')
            return

        # --- Decision node ---
        step = node['step']
        pool = node.get('pool', 0)
        pool_str = node.get('pool_str', '{}')
        cleared = node.get('cleared_str', '{}')

        lines = [f'Step {step}']
        if show_pool_labels:
            lines.append(f'test {pool_str}')
        lines.append(f'cleared={cleared}')
        if show_posteriors and node.get('posteriors'):
            lines.append(_posteriors_label(node['posteriors']))
        label = '\n'.join(lines)

        border = _border_color(pool)
        dot.node(nid, label=label, shape='box',
                 style='filled', fillcolor='#cce5ff',
                 color=border, penwidth='2')

        if parent_id is not None:
            dot.edge(parent_id, nid,
                     label=f' r={edge_label} ' if edge_label is not None else '')

        # Recurse into children
        for r in sorted(node.get('children', {}).keys()):
            _add_subtree(node['children'][r], parent_id=nid, edge_label=r)

    _add_subtree(tree)
    return dot


# ---------------------------------------------------------------------------
# Side-by-side and series rendering
# ---------------------------------------------------------------------------

def render_side_by_side(tree_a, tree_b, n, title_a="A", title_b="B", **kwargs):
    """Render two trees side by side using IPython HTML divs.

    Parameters
    ----------
    tree_a, tree_b : dict
        Trees from ``extract_tree()``.
    n : int
        Population size.
    title_a, title_b : str
        Labels for each tree.
    **kwargs
        Passed through to ``render_tree()``.

    Returns
    -------
    IPython.display.HTML or graphviz.Digraph
        HTML object with flex layout; falls back to a combined Digraph
        if IPython is not available.
    """
    dot_a = render_tree(tree_a, n, title=title_a, **kwargs)
    dot_b = render_tree(tree_b, n, title=title_b, **kwargs)

    try:
        from IPython.display import HTML
        svg_a = dot_a.pipe(format='svg').decode('utf-8')
        svg_b = dot_b.pipe(format='svg').decode('utf-8')
        html = (
            '<div style="display:flex; gap:20px; align-items:flex-start;">'
            f'<div><h3>{title_a}</h3>{svg_a}</div>'
            f'<div><h3>{title_b}</h3>{svg_b}</div>'
            '</div>'
        )
        return HTML(html)
    except ImportError:
        # Fallback: combined Digraph with subgraphs
        combined = graphviz.Digraph(format='svg')
        combined.attr(rankdir='TB')
        with combined.subgraph(name='cluster_a') as sa:
            sa.attr(label=title_a)
            sa.subgraph(dot_a)
        with combined.subgraph(name='cluster_b') as sb:
            sb.attr(label=title_b)
            sb.subgraph(dot_b)
        return combined


def render_tree_series(trees, n, titles, max_per_row=3, **kwargs):
    """Render a list of trees in a grid layout.

    Parameters
    ----------
    trees : list[dict]
        List of trees from ``extract_tree()``.
    n : int
        Population size.
    titles : list[str]
        Title for each tree.
    max_per_row : int
        Maximum trees per row.
    **kwargs
        Passed through to ``render_tree()``.

    Returns
    -------
    IPython.display.HTML or list[graphviz.Digraph]
        HTML object with flex-wrap layout; falls back to a list of Digraph
        objects if IPython is not available.
    """
    dots = [render_tree(t, n, title=ttl, **kwargs)
            for t, ttl in zip(trees, titles)]

    try:
        from IPython.display import HTML
        items = []
        for dot, ttl in zip(dots, titles):
            svg = dot.pipe(format='svg').decode('utf-8')
            items.append(
                f'<div style="flex:0 0 {100 // max_per_row}%; '
                f'max-width:{100 // max_per_row}%; box-sizing:border-box; '
                f'padding:5px;">'
                f'<h3>{ttl}</h3>{svg}</div>'
            )
        html = (
            '<div style="display:flex; flex-wrap:wrap; '
            'align-items:flex-start;">'
            + ''.join(items) +
            '</div>'
        )
        return HTML(html)
    except ImportError:
        return dots
