# VW and adaptive submodularity — open theoretical question

Companion to `augmented/vw_demo.py`. Lays out the precise question that
must be answered before the VW super-node framework yields a global
α-approximation guarantee for Greedy Dynamic.

## Setup

- Population N, budget B, max pool size G, prior p, utility u(F, Z).
- After k tests with covered set S = ∪ pools and history h_k, V = N \ S.
- VW instance at step k+1: super-nodes W = {w_T : T ⊆ S}; pool t = U ∪ T
  with U ⊆ V, T ⊆ S, |U|+|T| ≤ G.
- Per-step value (proper, contextual):
  ```
  F_h(T, U)  =  E[ u(h ⊕ (T∪U, r)) − u(h)  |  history h ]
              =  Σ_r  Pr(r | T∪U, h) · ( u(h ⊕ (T∪U, r)) − u(h) )
  ```
  where r = |Z ∩ (T∪U)| and the expectation is over Z conditioned on h.
- Cumulative greedy value over B steps:
  ```
  V_greedy(p, B, G)  =  E_{traj} Σ_{k=0}^{B-1} F_{h_k}(T_{k+1}, U_{k+1})
  ```

## What "scalar VW" gives empirically (from the demo)

- prob_A = ∏_{i∈T}(1−p_i), util_T = Σ u_i, weight = |T| → scalar VW
  reproduces the standard myopic-greedy step value EXACTLY.
- prob_B = OR-event → strictly worse on this objective.
- Neither captures r > 0 information; both lose strictly to F_h(T, U) on
  multi-step instances. Demo: gap grows 0.14 (B=2) → 1.96 (B=3) on
  n=6, G=3.

## What an α-approximation per step buys (Golovin–Krause framework)

A function f : 2^X × Φ → R≥0 with adaptive monotonicity and adaptive
submodularity yields a (1 − e^{−1}) guarantee for adaptive greedy
selection (Golovin–Krause 2011).

**Adaptive monotonicity (AM):** for any partial realization ψ and any
e ∉ dom(ψ),  E[f(dom(ψ) ∪ {e}, Φ) | ψ] ≥ E[f(dom(ψ), Φ) | ψ].

**Adaptive submodularity (AS):** for any ψ ⊆ ψ' and e ∉ dom(ψ'),
  Δ(e | ψ) ≥ Δ(e | ψ'),  where Δ(e | ψ) = E[f(dom(ψ)∪{e}) − f(dom(ψ)) | ψ].

If both hold and the per-step greedy choice is α-approximate, then
adaptive greedy is (1 − e^{−α})-optimal.

## Where VW runs into trouble

1. **f is not "selection of items".** A pool is a set of size ≤ G picked
   in one shot, not items selected sequentially within a step. The
   Golovin–Krause framework selects single elements; pool selection is
   (per step) an instance of *cost-budgeted* selection. Per-step α may
   refer to weighted-knapsack approximation, not adaptive submodularity.

2. **F_h(T, U) is not separable.** The contraejemplo in the demo shows
   two T's with identical (OR, all-clear, util_T) but different count
   PMFs ⇒ different downstream Bayesian updates ⇒ different F_h(T, U)
   for any fixed U with at least one new test left. Therefore the
   scalar (weight, prob, util) compression is provably lossy in the
   multi-step setting.

3. **AS does not follow from per-step submodularity alone.** Even if the
   pool-utility u(F, Z) → "Σ u_i over cleared at end" is submodular as a
   set function (which it is: it's a coverage-type function with respect
   to "is i ever in a pool that returned r=0"), this submodularity is
   over *subsets of N*, not over the augmented (V, W) state. The
   posterior correlations introduced by augmented results r > 0 break
   the standard reductions.

## A clean question to ask

Define
```
g_h(T, U)  =  F_h(T, U)
           =  E[ Σ_{i ∈ cleared at end} u_i  |  h, action = T∪U at step k+1 ]
                                                  − E[ ... | h, action = ∅ ]
```
treating g_h as a function of the pool t = T ∪ U for fixed history h.

**Q1.** Is g_h adaptively monotone w.r.t. partial pool construction?
i.e., does adding a single individual i to a partially built pool t' ⊆ t
ever decrease the (conditional) expected future utility? Intuitively
yes for u = "Σ cleared" — adding to a pool can only widen the cleared
set if r=0, and when r>0 it provides extra information. But this needs
a clean proof.

**Q2.** Is g_h adaptively submodular as a *set-valued* action over a
single step? This is *not* the standard AS framework (which selects
items across steps). One reformulation: replace each step's pool
choice by G atomic "add-individual" choices — does the resulting
sequential problem satisfy AS?

**Q3.** If Q2 holds, what is the per-step approximation factor for
VW-style enumeration of W? Naive enumeration is exact (= myopic
optimum), so α = 1, but the complexity is 2^|S|. Is there a constant α
enumeration with poly(|S|) complexity?

**Q3 — empirical evidence (see `augmented/vw_restrict.py`,
`augmented/vw_restrict_sweep.py`).**

A second heuristic, **`partner`**, dominates `self_score` in 4 of 6
regimes and stays ≤ 3 in mean L_min everywhere. It is

  partner(T)  =  (Σ u_T + u*) · ∏(1 − p_T) · p*

where (u*, p*) is the (utility, all-clear-prob) of the *globally* best
V-pool of size G − |T|, precomputed once. partner accounts for the
budget consumed by T (which self_score ignores), and on the
hand-crafted adversarial instance below it gives L_min = 2 vs L_min = 8
for self_score. Updated sweep:

| Regime                               | mean |W| | partner   | self      | ent_λ=1   | prob      |
|--------------------------------------|---------:|----------:|----------:|----------:|----------:|
| baseline (n=10, G=4)                 |     79.7 | **1.6/13** |  4.3/21   |  7.0/38   | 16.1/98   |
| high prevalence (p~U[.4,.7])         |     79.7 |  2.8/14   | **0.7/2** |  0.9/3    |  4.8/42   |
| larger n (n=15, G=5)                 |    278.3 | **1.1/4**  | 12.7/132  | 18.2/184  | 38.3/233  |
| deep history (n=10, G=3, 3-step)     |     62.9 | **1.1/3**  |  3.8/18   |  4.4/23   | 14.1/40   |
| bimodal p (10/90 mix)                |     79.7 | **1.0/1**  |  8.4/35   |  9.8/47   | 22.5/86   |
| outlier utility (u₀=50)              |     79.7 | **1.4/12** |  2.4/13   |  2.5/14   | 12.8/82   |

Why partner works: it is essentially a *lower bound* on val(T) using
the cheapest possible surrogate for U. It is tight whenever the
globally-best V-pool does not change with |T|, which holds in most of
our trials. In the high-prevalence regime self_score is already
near-trivial (mean L_min = 0.7) because the optimum often lives at
|T| ≤ 1, and partner's coarser bound adds a small constant overhead
without breaking anything (mean 2.8, max 14). Critical takeaway: mean
L_min stays near-constant as |W| grows from 79 to 278 — partner
**scales independently of |S|**.

Entropy-augmented `self + λ·H(r_T)` does NOT help — it consistently
matches or *worsens* self_score, because high-entropy T's are usually
suboptimal pools (large |T| with mid-range probabilities). Information
value is not the right augmentation for the myopic step.

**Adversarial worst-case (`adversarial_instance()`).** S = {0,1,2,3},
V = {4,5}, G = 4, p = [.01,.01,.6,.05, .05,.5], u = [1,1,100,50, 60,80].
Optimal pool = {3,4} (val = 99.27), but self_score ranks {2,3} top
because util_T = 150 dominates the small prob_T = 0.4. self_score
L_min = 8/15 = 53%; partner L_min = 2/15 = 13%; entropy variants 8–12.


Restrict W to top-L by a cheap ranking heuristic and measure L_min,
the smallest L such that the optimal T* is in the top L. K=20 trials
per regime, fmt mean / max:

| Regime                                 | mean |W| | self      | prob      | util       | random      |
|----------------------------------------|---------:|----------:|----------:|-----------:|------------:|
| baseline (n=10, G=4, p~U[.05,.35])     |     79.7 |  **4.3 / 21**  | 16.1 / 98 |  39.9 /155 |  40.5 /137  |
| high prevalence (p~U[.4,.7])           |     79.7 |  **0.7 /  2**  |  4.8 / 42 |  31.3 / 93 |  28.8 / 89  |
| larger n (n=15, G=5)                   |    278.3 | **12.7 /132**  | 38.3 /233 | 158.0 /363 | 128.6 /401  |
| deep history (n=10, G=3, 3-step)       |     62.9 |  **3.8 / 18**  | 14.1 / 40 |  19.1 / 66 |  33.9 / 86  |
| bimodal p (10/90 mix of 0.05/0.7)      |     79.7 |  **8.4 / 35**  | 22.5 / 86 |  28.6 / 75 |  39.8 /144  |
| outlier utility (u₀=50, rest U[1,10])  |     79.7 |  **2.4 / 13**  | 12.8 / 82 |  19.1 / 64 |  34.5 /121  |

**Why self_score works.** It is the exact value of "pool = T alone".
In the optimal pool's score `(Σ u_{T∪U}) · ∏(1 − p_{T∪U})`, the T-factors
enter additively in utility and multiplicatively in probability — the
same role as the U-factors — so a T that is itself a good atomic pool
is the most likely T-component of the optimum.

**Robustness verdict.** self_score wins across all six regimes. Mean
L_min / |W| ranges from 0.9% (high prevalence, where the optimum is
nearly always |T|=1 or |T|=2 of the highest-prob individual) to 10.5%
(bimodal prevalence, where ties between low-p and high-p clusters
break harder). Worst case remains a concern: max L_min can reach
~47% of |W| in larger-n trials. So self_score gives a strong
**average-case** restriction but **no worst-case bound**.

**What this resolves and what it does not.**
- ✅ Resolves: VW enumeration is computationally practical in
  expectation. Sort by self_score (O(|S|·2^|S|), but constant-time per
  candidate) and keep top L ≈ 0.05–0.15·|W|.
- ❌ Does not resolve: a worst-case bound on L_min. An adversarial
  instance where self_score requires Ω(|W|) is not ruled out — and
  the n=15 trial reaching L_min = 132 is suggestive evidence one exists.
- ❌ Does not address whether top-L self_score restriction preserves
  the α-approximation under lookahead. self_score ranks myopic value,
  but the demo's contraejemplo (T₁, T₂ with same self_score but
  different count-PMF entropy) shows multi-step information value is
  orthogonal. A lookahead-aware ranking — e.g., self_score augmented
  by H(r_T) — would be the natural extension to test next.

**Q4 — the killer question.** Even granting Q1–Q3, does the augmented
observation r ∈ {0, 1, …, |t|} preserve adaptive submodularity? The
Golovin–Krause results assume the realization Φ is observed exactly
(individual outcomes); a count is a coarsening, and the standard
reductions (Asadpour–Nazerzadeh, Chen–Krause) require care. We have
seen in the demo that count PMF differences exist even when scalar
summaries coincide — this is exactly where AS could break.

## Verdict and next step

The empirical demo establishes:
- VW scalar (with prob_A) ≡ myopic greedy. No algorithmic improvement.
- Closing the lookahead gap requires the count PMF of w_T, at which
  point the formulation is no longer "scalar super-node" but full pool
  enumeration with proper joint posterior — i.e., the existing DP.

For the framework to deliver theoretical value, **the right line of
attack is Q4**: prove (or refute) that the count-augmented adaptive
greedy on g_h satisfies Golovin–Krause AS. If yes, an (1 − e^{−α})
guarantee follows. If no, VW is at best a re-formulation, not a
guarantee-yielding reduction.

Concretely: the next paper-level task is to either (a) derive AS for
g_h with count observations under independent prior, or (b) construct
a counterexample where AS fails — a small instance where adaptive
greedy with VW enumeration does worse than (1 − 1/e) of OPT. The
demo's contraejemplo (T₁, T₂ with same OR but different PMF) is the
seed of such a counterexample if extended to a multi-step instance.
