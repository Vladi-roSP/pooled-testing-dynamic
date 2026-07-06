"""
CSV-based experiment runner with graph generation for augmented adaptive group counting.

Follows the same methodology as classical/solvers/ but for augmented strategies.
Generates random instances, saves to CSV, evaluates all strategies, and produces
comparison plots (box plots, violin plots, scatter plots).

Usage:
    python augmented/csv_experiments.py                    # default: n=5, B=2, G=3
    python augmented/csv_experiments.py --n 4 --B 2 --G 2  # custom parameters
    python augmented/csv_experiments.py --draws 200      # more draws

Output:
    augmented/data/results_N{n}_B{B}_G{G}.csv  — raw results
    augmented/figures/                          — comparison plots
"""

import sys
import os
import random
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from augmented.core import mask_from_indices, indices_from_mask, test_result
from augmented.baselines import u_max, u_single
from augmented.solver import solve_optimal_dapts
from augmented.classical_solver import solve_classical_dynamic
from augmented.greedy import (
    greedy_myopic_expected_utility,
    greedy_myopic_counting_expected_utility,
    greedy_myopic_gibbs_expected_utility,
)


# -------------------------------------------------------------------
# Data generation (matches classical methodology)
# -------------------------------------------------------------------

def create_agents(n, u_integers=False, seed=None):
    """Generate random agents with (id, utility, latent_state_probability).

    Matches the format from classical/solvers/greedyDynamicDraw.py.
    """
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        if u_integers:
            utility = rng.randint(1, 100)
        else:
            utility = rng.random()
        clearance_prob = rng.random()  # probability of being CLEARANCEY
        agents.append((i, utility, clearance_prob))
    return agents


def agents_to_p_u(agents):
    """Convert agents list to (p, u) vectors.

    agents[i] = (id, utility, clearance_probability)
    p[i] = 1 - clearance_probability (latent-state probability)
    u[i] = utility
    """
    p = [1.0 - a[2] for a in agents]
    u = [a[1] for a in agents]
    return p, u


def generate_clearance_status(agents, seed=None):
    """Generate a random clearance status vector based on agent probabilities.

    Returns list of 0/1 where 0 = active, 1 = clearancey.
    """
    rng = random.Random(seed)
    status = []
    for _, _, clearance_prob in agents:
        status.append(1 if rng.random() < clearance_prob else 0)
    return status


def generate_dataset(n, num_draws, B, G, u_integers=False, base_seed=42):
    """Generate a dataset of random instances.

    Returns a DataFrame with columns:
      - agents: string representation of agent tuples
      - clearanceStatus: string representation of clearance outcomes
      - p_i columns: individual latent-state probabilities
      - u_i columns: individual utilities
      - avg_p: average latent-state probability
    """
    rows = []
    for s in range(num_draws):
        seed_agents = base_seed + s
        seed_clearance = base_seed + num_draws + s

        agents = create_agents(n, u_integers=u_integers, seed=seed_agents)
        clearance_status = generate_clearance_status(agents, seed=seed_clearance)
        p, u = agents_to_p_u(agents)

        row = {
            'draw_id': s,
            'agents': str(agents),
            'clearanceStatus': str(clearance_status),
            'avg_p': sum(p) / len(p),
        }
        for i in range(n):
            row[f'p_{i}'] = p[i]
            row[f'u_{i}'] = u[i]

        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Strategy evaluation
# -------------------------------------------------------------------

def evaluate_row(row, n, B, G, include_optimal=True):
    """Evaluate all strategies for one instance.

    Returns dict of strategy -> expected utility.
    """
    p = [row[f'p_{i}'] for i in range(n)]
    u = [row[f'u_{i}'] for i in range(n)]

    results = {}

    # Baselines
    results['U_max'] = u_max(p, u)
    results['U_single'], _ = u_single(p, u, B)

    # Optimal solvers (expensive)
    if include_optimal and n <= 14:
        try:
            results['U_D_classical'], _ = solve_classical_dynamic(p, u, B, G)
        except Exception:
            pass
        try:
            results['U_D_A_optimal'], _ = solve_optimal_dapts(p, u, B, G)
        except Exception:
            pass

    # Greedy strategies
    results['U_greedy_sequential'] = greedy_myopic_expected_utility(p, u, B, G)
    results['U_greedy_counting'] = greedy_myopic_counting_expected_utility(p, u, B, G)

    return results


def run_all_evaluations(df, n, B, G, include_optimal=True):
    """Evaluate all strategies for every row in the dataset."""
    all_results = []
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Evaluating draw {idx + 1}/{total}...")

        results = evaluate_row(row, n, B, G, include_optimal=include_optimal)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    return pd.concat([df, results_df], axis=1)


# -------------------------------------------------------------------
# Graph generation
# -------------------------------------------------------------------

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    sns.set_style("whitegrid")


def plot_box_comparison(df, n, B, G, output_dir):
    """Box plots comparing all strategy performances."""
    setup_plot_style()

    strategy_cols = [c for c in df.columns if c.startswith('U_')]
    plot_df = df[strategy_cols].copy()

    # Rename for readability
    rename_map = {
        'U_max': 'U_max (upper bound)',
        'U_single': 'U_single (individual)',
        'U_D_classical': 'U_D (classical dynamic)',
        'U_D_A_optimal': 'U_DA (augmented optimal)',
        'U_greedy_sequential': 'U_greedy (sequential Bayes)',
        'U_greedy_counting': 'U_greedy (counting Bayes)',
    }
    plot_df = plot_df.rename(columns={k: v for k, v in rename_map.items()
                                       if k in plot_df.columns})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, orient='h', ax=ax, palette='viridis')
    ax.set_title(f"Strategy Performance Comparison\nn={n}, B={B}, G={G} "
                 f"({len(df)} draws)")
    ax.set_xlabel("Expected Utility")
    plt.tight_layout()
    path = os.path.join(output_dir, f'boxplot_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_violin_comparison(df, n, B, G, output_dir):
    """Violin plots showing distribution of each strategy."""
    setup_plot_style()

    strategy_cols = [c for c in df.columns if c.startswith('U_')]
    plot_df = df[strategy_cols].copy()

    rename_map = {
        'U_max': 'U_max',
        'U_single': 'U_single',
        'U_D_classical': 'U_D (classical)',
        'U_D_A_optimal': 'U_DA (augmented)',
        'U_greedy_sequential': 'Greedy (seq)',
        'U_greedy_counting': 'Greedy (count)',
    }
    plot_df = plot_df.rename(columns={k: v for k, v in rename_map.items()
                                       if k in plot_df.columns})

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(data=plot_df, orient='h', inner='box',
                   linewidth=1.2, ax=ax, palette='Set2')
    ax.set_title(f"Strategy Performance Distributions\nn={n}, B={B}, G={G} "
                 f"({len(df)} draws)")
    ax.set_xlabel("Expected Utility")
    plt.tight_layout()
    path = os.path.join(output_dir, f'violin_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_augmented_benefit(df, n, B, G, output_dir):
    """Scatter plot: augmented benefit over classical vs avg latent-state rate."""
    if 'U_D_classical' not in df.columns or 'U_D_A_optimal' not in df.columns:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    benefit = ((df['U_D_A_optimal'] - df['U_D_classical'])
               / df['U_D_classical'].clip(lower=1e-10) * 100)

    sc = ax.scatter(df['avg_p'], benefit, c=benefit,
                    cmap='RdYlGn', s=50, alpha=0.7, edgecolors='gray',
                    linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Augmented Benefit (%)')

    ax.set_xlabel('Average Latent-State Probability')
    ax.set_ylabel('Augmented Benefit over Classical (%)')
    ax.set_title(f"Augmented vs Classical: Benefit by LatentState Rate\n"
                 f"n={n}, B={B}, G={G} ({len(df)} draws)")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, f'augmented_benefit_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_greedy_vs_optimal(df, n, B, G, output_dir):
    """Scatter plot: greedy strategies vs optimal."""
    if 'U_D_A_optimal' not in df.columns:
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (greedy_col, label) in enumerate([
        ('U_greedy_sequential', 'Greedy (Sequential Bayes)'),
        ('U_greedy_counting', 'Greedy (Counting Bayes)'),
    ]):
        ax = axes[i]
        optimal = df['U_D_A_optimal']
        greedy = df[greedy_col]

        ax.scatter(optimal, greedy, s=30, alpha=0.6, color='steelblue',
                   edgecolors='gray', linewidth=0.3)

        # 45-degree line
        lims = [min(optimal.min(), greedy.min()),
                max(optimal.max(), greedy.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='optimal = greedy')

        gap = ((optimal - greedy) / optimal.clip(lower=1e-10) * 100)
        ax.set_xlabel('Optimal (U_DA)')
        ax.set_ylabel(label)
        ax.set_title(f"{label}\nMean gap: {gap.mean():.2f}%")
        ax.legend()

    fig.suptitle(f"Greedy vs Optimal Comparison\nn={n}, B={B}, G={G} "
                 f"({len(df)} draws)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'greedy_vs_optimal_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sequential_vs_counting(df, n, B, G, output_dir):
    """Scatter plot: sequential greedy vs counting greedy."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    seq = df['U_greedy_sequential']
    cnt = df['U_greedy_counting']

    ax.scatter(seq, cnt, s=40, alpha=0.6, c=df['avg_p'],
               cmap='coolwarm', edgecolors='gray', linewidth=0.3)

    lims = [min(seq.min(), cnt.min()), max(seq.max(), cnt.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='equal')

    ax.set_xlabel('Greedy (Sequential Bayesian)')
    ax.set_ylabel('Greedy (Full-History Counting)')
    ax.set_title(f"Sequential vs Counting Bayesian Greedy\n"
                 f"n={n}, B={B}, G={G} ({len(df)} draws)")
    ax.legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(df['avg_p'].min(),
                                                   df['avg_p'].max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Avg Latent-State Probability')

    plt.tight_layout()
    path = os.path.join(output_dir, f'seq_vs_counting_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_performance_by_latent_state_rate(df, n, B, G, output_dir):
    """Line plot: strategy performance vs latent-state rate bins."""
    setup_plot_style()

    # Bin by average latent-state rate
    df = df.copy()
    df['p_bin'] = pd.cut(df['avg_p'], bins=5, labels=False)
    bin_edges = pd.cut(df['avg_p'], bins=5, retbins=True)[1]
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                  for i in range(len(bin_edges) - 1)]

    strategy_cols = [c for c in df.columns
                     if c.startswith('U_') and c != 'U_max']

    rename_map = {
        'U_single': 'Individual',
        'U_D_classical': 'Classical DP',
        'U_D_A_optimal': 'Augmented DP',
        'U_greedy_sequential': 'Greedy (seq)',
        'U_greedy_counting': 'Greedy (count)',
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for col in strategy_cols:
        if col in rename_map:
            means = df.groupby('p_bin')[col].mean()
            ax.plot(range(len(means)), means.values, 'o-',
                    label=rename_map[col], linewidth=2, markersize=6)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_xlabel('Average Latent-State Probability Range')
    ax.set_ylabel('Mean Expected Utility')
    ax.set_title(f"Strategy Performance by LatentState Rate\n"
                 f"n={n}, B={B}, G={G} ({len(df)} draws)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, f'performance_by_p_N{n}_B{B}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_plots(df, n, B, G, output_dir):
    """Generate all comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating plots...")

    plot_box_comparison(df, n, B, G, output_dir)
    plot_violin_comparison(df, n, B, G, output_dir)
    plot_augmented_benefit(df, n, B, G, output_dir)
    plot_greedy_vs_optimal(df, n, B, G, output_dir)
    plot_sequential_vs_counting(df, n, B, G, output_dir)
    plot_performance_by_latent_state_rate(df, n, B, G, output_dir)


# -------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------

def print_summary(df, n, B, G):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY: n={n}, B={B}, G={G}, {len(df)} draws")
    print(f"{'='*70}")

    strategy_cols = [c for c in df.columns if c.startswith('U_')]
    for col in sorted(strategy_cols):
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:30s}  mean={vals.mean():.4f}  "
                  f"std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

    # Augmented benefit
    if 'U_D_classical' in df.columns and 'U_D_A_optimal' in df.columns:
        benefit = ((df['U_D_A_optimal'] - df['U_D_classical'])
                   / df['U_D_classical'].clip(lower=1e-10) * 100)
        print(f"\n  Augmented benefit over classical:")
        print(f"    mean = +{benefit.mean():.2f}%,  "
              f"max = +{benefit.max():.2f}%,  "
              f"median = +{benefit.median():.2f}%")

    # Counting vs sequential greedy
    diff = df['U_greedy_counting'] - df['U_greedy_sequential']
    print(f"\n  Counting vs Sequential greedy difference:")
    print(f"    mean = {diff.mean():.6f},  "
          f"std = {diff.std():.6f},  "
          f"max abs = {diff.abs().max():.6f}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Item 1: B >= 3 experiments — divergence between greedy variants
# -------------------------------------------------------------------

def generate_dataset_p_range(n, num_draws, p_range=(0.0, 1.0),
                              u_integers=False, base_seed=42):
    """Generate dataset with latent-state probabilities in a given range."""
    rng = random.Random(base_seed)
    rows = []
    for s in range(num_draws):
        p_lo, p_hi = p_range
        p = [rng.uniform(p_lo, p_hi) for _ in range(n)]
        if u_integers:
            u = [rng.randint(1, 100) for _ in range(n)]
        else:
            u = [rng.uniform(0.5, 5.0) for _ in range(n)]
        row = {'draw_id': s, 'avg_p': sum(p) / len(p)}
        for i in range(n):
            row[f'p_{i}'] = p[i]
            row[f'u_{i}'] = u[i]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_row_greedy_variants(row, n, B, G):
    """Evaluate sequential, counting, and gibbs greedy for one instance."""
    p = [row[f'p_{i}'] for i in range(n)]
    u = [row[f'u_{i}'] for i in range(n)]
    results = {}
    results['U_greedy_sequential'] = greedy_myopic_expected_utility(p, u, B, G)
    results['U_greedy_counting'] = greedy_myopic_counting_expected_utility(p, u, B, G)
    results['U_greedy_gibbs'] = greedy_myopic_gibbs_expected_utility(
        p, u, B, G, seed=42)
    if n <= 14:
        try:
            results['U_D_A_optimal'], _ = solve_optimal_dapts(p, u, B, G)
            results['U_D_classical'], _ = solve_classical_dynamic(p, u, B, G)
        except Exception:
            pass
    return results


def run_b_comparison(n=5, B_values=None, G=3, num_draws=100,
                     p_range=(0.1, 0.5), seed=42):
    """Run experiments across multiple B values to show greedy divergence.

    Returns a DataFrame with results for each (B, draw_id).
    """
    if B_values is None:
        B_values = [2, 3, 4, 5]

    all_rows = []
    df_base = generate_dataset_p_range(n, num_draws, p_range=p_range,
                                        base_seed=seed)

    for B in B_values:
        print(f"\n  B={B}: evaluating {num_draws} instances (n={n}, G={G})...")
        for idx, row in df_base.iterrows():
            if (idx + 1) % 25 == 0:
                print(f"    draw {idx + 1}/{num_draws}")
            results = evaluate_row_greedy_variants(row, n, B, G)
            result_row = dict(row)
            result_row['B'] = B
            result_row.update(results)

            # Compute divergence
            result_row['seq_count_diff'] = (results['U_greedy_counting']
                                             - results['U_greedy_sequential'])
            all_rows.append(result_row)

    return pd.DataFrame(all_rows)


def plot_b_divergence(df, n, G, output_dir):
    """Plot divergence between sequential and counting greedy across B values."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot of divergence by B
    ax = axes[0]
    b_values = sorted(df['B'].unique())
    data_by_b = [df[df['B'] == b]['seq_count_diff'].values for b in b_values]
    bp = ax.boxplot(data_by_b, labels=[str(b) for b in b_values],
                    patch_artist=True)
    colors = plt.cm.viridis([i / len(b_values) for i in range(len(b_values))])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Test Budget B')
    ax.set_ylabel('Counting − Sequential EU')
    ax.set_title('Greedy Divergence Grows with B')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Mean divergence by B
    ax = axes[1]
    means = df.groupby('B')['seq_count_diff'].agg(['mean', 'std', 'max'])
    ax.bar(range(len(means)), means['mean'], yerr=means['std'],
           color='steelblue', edgecolor='black', capsize=5, alpha=0.8)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([str(b) for b in means.index])
    ax.set_xlabel('Test Budget B')
    ax.set_ylabel('Mean Divergence (Counting − Sequential)')
    ax.set_title('Cross-Test Information Increases with B')

    # Add max annotation
    for i, (b, row) in enumerate(means.iterrows()):
        ax.annotate(f"max={row['max']:.4f}", (i, row['mean'] + row['std']),
                    ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'Sequential vs Counting Greedy Divergence (n={n}, G={G})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'b_divergence_N{n}_G{G}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return means


# -------------------------------------------------------------------
# Item 3: High latent-state rate experiments at scale
# -------------------------------------------------------------------

def run_high_latent_state_experiment(n_values=None, B_values=None, G=3,
                                  num_draws=200, seed=42):
    """Run experiments with high latent-state rates to confirm augmented benefit.

    Tests Francisco's hypothesis: augmented benefit increases with latent-state rate.
    Returns a DataFrame with results across latent-state rate regimes.
    """
    if n_values is None:
        n_values = [5]
    if B_values is None:
        B_values = [2, 3]

    regimes = [
        ('low', 0.05, 0.15),
        ('medium', 0.15, 0.35),
        ('high', 0.35, 0.55),
        ('very_high', 0.55, 0.80),
    ]

    all_rows = []
    for n in n_values:
        for B in B_values:
            for regime_name, p_lo, p_hi in regimes:
                print(f"\n  n={n}, B={B}, G={G}, regime={regime_name} "
                      f"(p∈[{p_lo},{p_hi}])...")
                df = generate_dataset_p_range(n, num_draws,
                                               p_range=(p_lo, p_hi),
                                               base_seed=seed)
                for idx, row in df.iterrows():
                    if (idx + 1) % 50 == 0:
                        print(f"    draw {idx + 1}/{num_draws}")
                    results = evaluate_row_greedy_variants(row, n, B, G)
                    result_row = dict(row)
                    result_row['n'] = n
                    result_row['B'] = B
                    result_row['regime'] = regime_name
                    result_row['p_lo'] = p_lo
                    result_row['p_hi'] = p_hi
                    result_row.update(results)

                    if 'U_D_A_optimal' in results and 'U_D_classical' in results:
                        ud = results['U_D_classical']
                        uda = results['U_D_A_optimal']
                        result_row['augmented_benefit'] = (
                            (uda - ud) / max(ud, 1e-10) * 100)
                    all_rows.append(result_row)

    return pd.DataFrame(all_rows)


def plot_high_latent_state_results(df, output_dir):
    """Plot augmented benefit by latent-state rate regime."""
    setup_plot_style()

    if 'augmented_benefit' not in df.columns:
        print("  No augmented_benefit column — skipping high latent_state plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot by regime
    ax = axes[0]
    regime_order = ['low', 'medium', 'high', 'very_high']
    regimes_present = [r for r in regime_order if r in df['regime'].values]
    data_by_regime = [df[df['regime'] == r]['augmented_benefit'].dropna().values
                      for r in regimes_present]
    bp = ax.boxplot(data_by_regime, labels=regimes_present, patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for patch, color in zip(bp['boxes'], colors[:len(regimes_present)]):
        patch.set_facecolor(color)
    ax.set_xlabel('LatentState Rate Regime')
    ax.set_ylabel('Augmented Benefit (%)')
    ax.set_title('Augmented Benefit by LatentState Regime')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Scatter: avg_p vs benefit with trend
    ax = axes[1]
    valid = df.dropna(subset=['augmented_benefit'])
    import numpy as np
    sc = ax.scatter(valid['avg_p'], valid['augmented_benefit'],
                    c=valid['avg_p'], cmap='RdYlBu_r', s=20, alpha=0.5)
    # Polynomial trend
    if len(valid) > 10:
        z = np.polyfit(valid['avg_p'], valid['augmented_benefit'], 2)
        trend = np.poly1d(z)
        x_trend = np.linspace(valid['avg_p'].min(), valid['avg_p'].max(), 100)
        ax.plot(x_trend, trend(x_trend), 'r-', linewidth=2,
                label=f'Trend (quadratic)')
        ax.legend()
    ax.set_xlabel('Average Latent-State Probability')
    ax.set_ylabel('Augmented Benefit (%)')
    ax.set_title('Augmented Benefit vs LatentState Rate')
    plt.colorbar(sc, ax=ax, label='Avg p')

    fig.suptitle('High LatentState Rate Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'high_latent_state_analysis.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary
    print("\n  Augmented benefit by regime:")
    for regime in regimes_present:
        data = df[df['regime'] == regime]['augmented_benefit'].dropna()
        if len(data) > 0:
            print(f"    {regime:12s}: mean={data.mean():+.3f}%, "
                  f"max={data.max():+.3f}%, n={len(data)}")


def main():
    parser = argparse.ArgumentParser(
        description='CSV-based experiments for augmented adaptive group counting')
    parser.add_argument('--n', type=int, default=5, help='Population size')
    parser.add_argument('--B', type=int, default=2, help='Test budget')
    parser.add_argument('--G', type=int, default=3, help='Max pool size')
    parser.add_argument('--draws', type=int, default=100,
                        help='Number of random instances')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--no-optimal', action='store_true',
                        help='Skip optimal solvers (faster)')
    parser.add_argument('--u-integers', action='store_true',
                        help='Use integer utilities (1-100)')
    args = parser.parse_args()

    n, B, G = args.n, args.B, args.G

    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Augmented Adaptive Group Counting: CSV Experiments")
    print(f"  n={n}, B={B}, G={G}, draws={args.draws}, seed={args.seed}")
    print()

    # Step 1: Generate dataset
    print("Step 1: Generating random instances...")
    t0 = time.time()
    df = generate_dataset(n, args.draws, B, G,
                          u_integers=args.u_integers, base_seed=args.seed)
    print(f"  Generated {len(df)} instances in {time.time() - t0:.1f}s")

    # Step 2: Evaluate all strategies
    print("\nStep 2: Evaluating strategies...")
    t0 = time.time()
    df = run_all_evaluations(df, n, B, G,
                             include_optimal=not args.no_optimal)
    elapsed = time.time() - t0
    print(f"  Evaluated all strategies in {elapsed:.1f}s")

    # Step 3: Save CSV
    csv_path = os.path.join(data_dir, f'results_N{n}_B{B}_G{G}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved results to: {csv_path}")

    # Step 4: Print summary
    print_summary(df, n, B, G)

    # Step 5: Generate plots
    generate_all_plots(df, n, B, G, fig_dir)

    print(f"\nDone! Results in {data_dir}, figures in {fig_dir}")


if __name__ == "__main__":
    main()
