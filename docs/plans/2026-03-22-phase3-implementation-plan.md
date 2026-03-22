# Phase 3: Implementation Plan — Gibbs Verification, Beta Reward & Scaled Experiments

**Date:** 2026-03-22
**Based on:** Meetings with Francisco (March 2026)
**Status:** Ready to implement

---

## Overview

Recent commits already delivered: hybrid greedy→DP solver, MOSEK/Gurobi pool selection, tree visualizer, and overnight experiment infrastructure. This plan covers what remains:

1. **Gibbs Verification** — Prove Gibbs sampling is correct (CRITICAL)
2. **Beta Meta-parameter** — Infection discovery reward for greedy
3. **Scaled Experiments** — Large-n runs with VIP scenarios and utility modulation
4. **Paper Examples** — Hand-computable examples and trade-off analysis

---

## What Already Exists

| Component | File | Status |
|-----------|------|--------|
| Hybrid greedy→DP solver | `hybrid_solver.py` | Done |
| MOSEK/Gurobi pool selection | `pool_solvers.py` | Done |
| Tree visualizer (Graphviz) | `tree_visualizer.py` | Done |
| Overnight experiment runner | `overnight_experiments.py` | Done |
| Infection-aware scoring (alpha) | `hybrid_solver.py:infection_aware_score` | Done |
| pool_selector in greedy | `greedy.py` (updated) | Done |
| First overnight results | `results/overnight_2026-03-21_215231.csv` | Done (n=10,20) |
| Semi-utility (alpha) | `semi_utility.py` | Done |
| Gibbs sampling | `bayesian.py:gibbs_update` | Done (9 basic tests) |

---

## SPRINT 1: Gibbs Sampling Verification (CRITICAL — DO FIRST)

Francisco emphasized this repeatedly: "Asegurarnos que el muestreo de Gibbs es aproximadamente correcto."

### 1.1 Systematic Exact vs Gibbs Comparison
**File:** Add to `tests.py`

For each config (n, B, G) in {(5,2,3), (5,3,3), (6,2,3), (6,3,4), (7,2,4), (7,3,4), (8,2,4), (8,3,5)}:
- Generate 50 random instances
- For each: create a random history by simulating greedy on a random z_mask
- Compare `bayesian_update_by_counting(p, history, n)` vs `gibbs_update(p, history, n)`
- **Pass criteria:** max |exact - gibbs| < 0.03 across all marginals, all instances

### 1.2 Gibbs Greedy vs Counting Greedy
**File:** Add to `tests.py`

For n=5,6,7,8 with B=2,3 and G=3:
- Compare `greedy_myopic_gibbs_expected_utility` vs `greedy_myopic_counting_expected_utility`
- **Pass criteria:** |EU_gibbs - EU_counting| / EU_counting < 1%

### 1.3 Hand-Computable Verification Examples
**File:** New `augmented/verification_examples.py`

Create 2-3 small examples (n=3,4) where posteriors are computed by hand:
- Example 1: n=3, history=((0b011, 1),) → compute P(Z_i=1) analytically
- Example 2: n=4, history=((0b0111, 0), (0b1100, 1)) → cross-test deduction
- Print step-by-step: prior → consistent worlds → posterior (exact) → Gibbs estimate
- These examples go into the Overleaf paper

### 1.4 Convergence Analysis
**File:** New `augmented/gibbs_analysis.py`

For a fixed instance (n=8, B=3, G=4):
- Run Gibbs with iterations in {100, 200, 500, 1000, 2000, 5000}
- Plot max|gibbs - exact| vs iterations
- Determine minimum iterations for reliable results
- Test with different n values to see scaling

---

## SPRINT 2: Infection Discovery Reward (Beta Meta-parameter)

### 2.1 Core Concept

The existing `infection_aware_score` in hybrid_solver.py blends myopic clearing gain with information gain using alpha. But Francisco wants a **separate, simpler mechanism** for the standard greedy:

> "Agregar un metaparametro a greedy en forma de premio para encontrar a personas infectadas, puedes agregale un pseudopremio. Es en esencia darle una manera indirecta de pensar a futuro."

### 2.2 How it differs from existing mechanisms

| Mechanism | File | What it does |
|-----------|------|-------------|
| Semi-utility (alpha) | `semi_utility.py` | Values partial health confidence |
| Infection-aware (alpha) | `hybrid_solver.py` | Blends myopic + info gain in hybrid solver |
| **Beta reward (NEW)** | **`infection_reward_greedy.py`** | **Pseudo-reward for confirming infections in standard greedy** |

Beta is specifically about: "If I test this pool and find r > 0, I learn WHO is infected and can stop wasting tests on them."

### 2.3 Implementation
**File:** New `augmented/infection_reward_greedy.py`

```python
def greedy_myopic_beta_simulate(p, u, B, G, z_mask, beta, info_metric='entropy')
def greedy_myopic_beta_expected_utility(p, u, B, G, beta, info_metric='entropy')
```

Info metrics:
- `'confirmed'`: count of individuals whose p crosses 0.99 or 0.01 threshold
- `'entropy'`: reduction in sum of binary entropies H(p_i)
- `'variance'`: reduction in sum of p_i*(1-p_i)

### 2.4 VIP Benchmark Instance
**File:** Add to `experiments.py` or new `augmented/vip_experiments.py`

Francisco's professors example formalized:
```python
# VIP: 8 professors, high infection, very high utility
p_vip = [0.8] * 8;    u_vip = [10.0] * 8
# Regular: 12 others, moderate infection, lower utility
p_reg = [0.2] * 12;   u_reg = [2.0] * 12
# Combined: n=20, B=6, G=10
```

Expected: beta-greedy should triage VIPs first, then redirect budget.

### 2.5 Beta Sweep
- beta in {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
- 50 instances per beta, across VIP and uniform populations
- Find optimal beta per regime

---

## SPRINT 3: Large-Scale Experiments

### 3.1 Run overnight_experiments.py at Scale
Already have infrastructure. Need to run:

| Config | n | B | G | Greedy | Notes |
|--------|---|---|---|--------|-------|
| A | 20 | 5 | 10 | Gurobi + Gibbs | Two big pools |
| B | 30 | 5 | 10 | Gurobi + Gibbs | Medium scale |
| C | 50 | 10 | 10 | Gurobi + Gibbs | Large scale |
| D | 20 | 2 | 10 | All variants | Francisco's "two big tests" |

Each with 50 instances across low/medium/high regimes.

### 3.2 VIP/Regular Mixed Population Experiments
- n=20 (8 VIP + 12 regular), B=6, G=10
- n=50 (15 VIP + 35 regular), B=10, G=15
- Compare: standard greedy vs beta-greedy vs semi-utility greedy
- **Key question:** Does beta outperform standard greedy on these?

### 3.3 Utility Modulation ("modular utilidades")
Same n,B,G but vary utility distributions:
- Uniform: u_i = 1
- Skewed: u_i in {1, 5, 10}
- Extreme: u_i in {1, 100}
- **Question:** When does augmented benefit most from utility structure?

### 3.4 Large G Exploration (Francisco's suggestion)
- n=20, B=2, G=10 — "two big pools"
- **Question:** Is this feasible? Does large G give more maneuverability?
- "Mientras más grande es el G nos da más margen de maniobra"

### 3.5 Nick's Thesis Cases
- Review Nick's thesis for cases where classical greedy performed well
- Re-run with augmented tests
- Compare augmented advantage in those specific regimes

---

## SPRINT 4: Paper Examples & Trade-Off Analysis

### 4.1 Hand-Computable Examples for Overleaf
Francisco: "Extraer ejemplos directos y algunos concretos, ponerlos en el Overleaf para hacer el cómputo a mano y asegurarnos que está bien hecho."

- **Example 1:** n=3, B=2, G=2 — full decision tree by hand
- **Example 2:** n=4, B=2, G=3 — VIP mini-scenario (p=(0.8,0.8,0.2,0.2), u=(10,10,2,2))
- **Example 3:** n=4, B=3, G=2 — sequential vs counting divergence by hand

### 4.2 Trade-Off Analysis
- **B vs G:** For fixed capacity B*G, few big pools vs many small?
- **Marginal value of B:** How much does each additional test help?
- **Where augmented shines:** Map the (n, B, G, p_avg) space

### 4.3 Hybrid Greedy Exploration (already implemented)
- Use `hybrid_solver.py` to explore K values
- "Ver la propuesta de tener un greedy híbrido que al final se haga brute force"
- Already done — need experiments comparing K=0 (full DP) vs K=1,2,3 vs K=B (full greedy)

---

## Implementation Order

| Sprint | Tasks | Depends on |
|--------|-------|-----------|
| **1** | Gibbs verification (1.1–1.4) | Nothing — do first |
| **2** | Beta meta-parameter (2.1–2.5) | Sprint 1 (need verified Gibbs) |
| **3** | Large-n experiments (3.1–3.5) | Sprint 1 + Sprint 2 |
| **4** | Paper examples + trade-offs (4.1–4.3) | Sprint 3 results |

---

## Key Questions This Phase Answers

1. **Is Gibbs correct?** → Systematic verification against exact counting
2. **Does beta help?** → VIP experiment: infection reward greedy vs standard
3. **Where does augmented shine at scale?** → Large-n across regimes
4. **What are the trade-offs?** → B vs G vs utility structure exploration
5. **Can we compute examples by hand?** → Verification for the paper
