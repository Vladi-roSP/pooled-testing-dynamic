# Claim audit for the unified separation-first paper

## Scope and status convention

This audit cross-checks the two English drafts against the executed post-session
notebook, its builder, and the relevant CSV artifacts. It is intentionally
conservative: a statement appearing in a draft is not treated as established
unless the supplied derivation, executed output, or data artifact supports it.

- **verified**: supported by a complete argument in the supplied sources or by an
  executed exact computation with a matching artifact.
- **qualified**: the core statement has support, but its hypotheses, scope, or
  numerical provenance must be stated more narrowly.
- **pending**: a requested comparison or general algorithm is not established in
  the supplied sources.
- **appendix/future**: even where there is partial evidence, the editorial decision
  is to keep the material outside the main line; unresolved proof or validation
  issues are recorded explicitly.

## Claim inventory

| Claim | Source | Evidence status | Destination |
|---|---|---|---|
| **Homogeneous static optimum:** for an infinite homogeneous population with utility \(u\), healthy probability \(q<1/2\), \(B\) tests, and hard clearing, the best fixed binary plan has \(U^{\mathrm{stat}}=Buq\). Individual tests attain the bound. For a pool of size \(g\), expected credited utility is \(ugq^g\le uq\); summing this per-test bound also covers overlapping fixed designs by a union bound. This is not a statement for heterogeneous \((q_i,u_i)\), finite-population exhaustion, or another reward convention. | `build_avances_post_sesion_notebook.py:60-62,80-105`; same markdown and computation in `avances_post_sesion.ipynb` | **verified** | Main separation theorem/lemma, with all hypotheses in the statement. |
| **Constructive dynamic-count lower bound:** with \(k\) disjoint initial pools of size \(G\) and a count-guided binary search, the event that at least one of the \(kG\) people is healthy has probability \(1-(1-q)^{kG}\). Under a convention that deductive identification earns utility, this gives \(U^{\mathrm{dyn,count}}\ge u[1-(1-q)^{kG}]\) with \(B=k+\log_2G\). Under the drafts' strict hard-clearing definition (utility only after membership in an observed zero-count pool), the final healthy singleton can be inferred without itself being tested, so one additional test is needed. With total budget \(B\), the safe strict-clearing form is therefore \(k=B-\log_2G-1\). For the anchor \((q,G,B)=(0.1,16,6)\), the notebook's reported \(0.966u\) uses the deductive-credit convention; strict clearing with the same total budget has \(k=1\) and the still-separating lower bound \(1-0.9^{16}\approx0.815u>0.6u\). When \(G\) is restricted to powers of two, the optimized coverage is \(kG^*=2^{B-1}\) under deductive credit but \(kG^*=2^{B-2}\) under strict clearing (attained, for example, by \(k=1,G=2^{B-2}\), with the adjacent \(k=2\) choice tying). | `build_avances_post_sesion_notebook.py:64-75,106-126,128-143,194-207`; executed notebook cell 3 reports `0.600`, `0.966`, `+61%`; strict reward definition in `dynamic_augmented_group_counting.tex:101-109` and `resolution_three_directions.tex:81-88` | **qualified** | Main separation result only after choosing one convention and correcting the budget/exponent consistently. Put the optimized-\(G\) refinement in the appendix. |
| **Intermediate cell: optimal dynamic binary policy** on the same homogeneous separation family, needed to decompose the gain into “dynamic” and “count-valued” components. | `build_avances_post_sesion_notebook.py:452-465` explicitly labels this cell pending; `dynamic_augmented_group_counting.tex:111-130` defines \(\mathcal U^D\) but does not solve the post-session family. | **pending** | A required main-text comparison/table cell. Until solved, describe static-vs-dynamic-count only as a separation that changes two features, not as the isolated value of augmentation. |
| **Monotonicity by channel refinement:** if \(Q'\) refines \(Q\), all channels used for hard clearing isolate \(\{0\}\), and policies may discard information, then \(\mathcal U_Q\le\mathcal U_{Q'}\). The stagewise deterministic-garbling proof is complete under those hypotheses. | `resolution_three_directions.tex:90-138`; implementation guards and endpoint equivalences in `augmented/solver.py:16-27`, `augmented/tests_resolution.py:14-27,41-58,103-122` | **verified** | Short supporting proposition in the theory section or appendix; it should not displace the separation result as the paper's anchor. |
| **Disjoint tested pools:** exact-count constraints factorize by disconnected components, so exact marginals can be computed componentwise; when later admissible actions concern disjoint people, the count contains no cross-pool information unavailable to the bit. The stronger sentence “the greedy equals the optimum” is not established for an unrestricted online choice among competing pools, because choosing a pool can block alternative packings; it is safe only for a fixed disjoint partition/action family. | `dynamic_augmented_group_counting.tex:173-186`; `resolution_three_directions.tex:216-226`; component decomposition and exact component enumeration in `augmented/bayesian.py:254-257,304-381,648-664` | **qualified** | First tractable regime in the main algorithm section; omit or sharply qualify “greedy equals optimum.” |
| **Laminar/nested pools:** for the fully observed nested example, subtracting parent/child counts produces independent layers, and the elementary-symmetric-polynomial algorithm computes exact marginals. The executed notebook matches brute force at \(n=12\) to \(2.2\times10^{-16}\) and processes the constructed \(n=6000\) three-layer instance in 13 ms. A general DP for partially observed laminar trees is described but not implemented or proved in the supplied sources. | `build_avances_post_sesion_notebook.py:218-313`; executed notebook cells 9-10; general draft claim in `dynamic_augmented_group_counting.tex:179-180` | **qualified** | Main algorithm/evidence section for the fully observed-layer case; general laminar-tree algorithm must be stated as pending unless completed. |
| **One-bit-separator / acyclic factor-graph chain:** the forward-backward routine over the single shared boundary variable is exact for pools \(\{0,1,2\},\{2,3,4\},\ldots\). The bipartite factor graph is an acyclic chain and each message crosses a one-bit separator. This should not be called “treewidth one” without qualification: the corresponding primal graph makes each three-person pool a clique and has treewidth \(2\). The routine matches brute force at \(n=13\) to \(1.1\times10^{-16}\) and runs on the constructed 200-pool, \(n=401\) instance in 1.1 ms. These are single-run notebook timings, not a benchmark distribution. | `build_avances_post_sesion_notebook.py:315-398`; executed notebook cell 12 | **verified** | Main tractable-algorithm example and reproducible empirical column, named by its one-bit separator or acyclic factor graph; timings remain illustrative. |
| **Bounded treewidth:** a junction-tree/variable-elimination method gives exact inference with cost exponential in the width and polynomial in instance size for fixed width. The drafts state this standard route, and the delivered chain routine realizes an acyclic factor-graph special case with one-bit messages; no general implementation, complexity statement with factor-domain dependence, or experiment is supplied. | `dynamic_augmented_group_counting.tex:181-183`; `build_avances_post_sesion_notebook.py:315-321,400-406` | **qualified** | State as the general algorithmic template; move any claim of a delivered general solver to future work unless implementation and tests are added. |
| **Few-test regime:** grouping individuals by their membership pattern across \(k\) tests is asserted to yield inference exponential in \(k\) and polynomial in \(N\). No derivation, exact complexity, implementation, or executed check is supplied, especially for heterogeneous priors. | `dynamic_augmented_group_counting.tex:184-185` | **pending** | Appendix/future work, or add a complete grouped-generating-function algorithm and proof before calling it a tractable delivered regime. |
| **Small-instance count-vs-binary gains:** the hierarchy experiment verifies \(+0.628407\%\) for \((N,B,G)=(3,2,3)\) over 200 instances, \(+3.969339\%\) for \((5,3,5)\) over 200, and \(+5.071847\%\) for \((7,3,7)\) over 40. These are the **means of the per-instance percentages** \(100(\mathcal U_A^D-\mathcal U^D)/\mathcal U^D\), computed in `hierarchy_experiment.py:58-81`; they are not percentages formed from the displayed mean welfare values. The corresponding ratios of means are \(0.670977\%\), \(3.711486\%\), and \(4.826598\%\), respectively. Raw CSV row counts and the summary `instances` fields agree, and all three summaries report zero hierarchy violations. The remaining textual inconsistency is that the experiment prose says 200 instances per configuration while the table caption correctly identifies 40 for the heavier \(N=7\) run. | `dynamic_augmented_group_counting.tex:254-280`; generator and estimator in `augmented/hierarchy_experiment.py:38-83`; raw artifacts `results/hierarchy/hierarchy_small.csv` and `results/hierarchy/hierarchy_n7.csv`; summaries `results/hierarchy/hierarchy_small_summary.csv` and `results/hierarchy/hierarchy_n7_summary.csv` | **verified** | Main empirical section. Label the last column explicitly as the mean per-instance relative gain, retain 200/200/40, and correct the prose sample-count sentence. |
| **Resolution curve:** on the supplied `mixta_n6_g4` instance, cap 1/2/3 values are \(4.967923,5.297456,5.357733\), so \(\{0,1,\ge2\}\) captures \(0.845368\) of the binary-to-count gap. The CSV matches the draft table, and all listed curves are nondecreasing. The evidence is a selected small-instance sweep, not a population-level result. For \(N=5,G=4\), equality of optimal values at caps 2-4 does not mean the channels literally coincide: counts 3 and 4 remain possible; only the optimal values coincide on those instances. “The only interior level” means only among the supplied instances. | `resolution_three_directions.tex:147-214`; `augmented/data/resolution_curve.csv`; endpoint/test scaffolding in `augmented/resolution_curve.py` and `augmented/tests_resolution.py` | **verified** | Empirical evidence in the main paper, with the scope and saturation explanation corrected; full sweep table may go to the appendix. |
| **\#P hardness of an exact posterior marginal:** the draft reduces from \#Exact Cover and correctly observes that the normalizing constant is a weighted count of feasible binary profiles. However, the written proof jumps from hardness of the normalizer \(Z\) to hardness of a single posterior marginal, which is a ratio and does not directly reveal \(Z\). A self-reduction/Turing-reduction argument or a claim restricted to computing \(Z\) is needed. | `dynamic_augmented_group_counting.tex:145-170`; notebook editorial instruction at `build_avances_post_sesion_notebook.py:461-462` | **appendix/future** | Appendix or future work only. Do not advertise the current marginal-hardness proof in the abstract or main contribution list until repaired. |
| **Exact fiber enumeration:** for pools \(\{0,1,2\}\) and \(\{2,3,4\}\), both observed with exact count one, exhaustive enumeration gives 5 count-consistent profiles versus 25 profiles under two positive binary outcomes; the all-active profile is excluded by exact counts and included by binary positives. | `build_avances_post_sesion_notebook.py:423-449`; executed notebook cell 16 | **appendix/future** | Reproducibility appendix or intuition box, not the main line. The finite enumeration itself is verified. |
| **Gibbs/fiber sampler:** componentwise enumeration is exact below the component cap, and swap-only moves preserve total active count, so the three-variable example can be trapped. The stronger claim that the implemented randomized alternating-path moves restore ergodicity for every exact-count fiber is not proved; arbitrary binary fibers generally require a demonstrated connecting move set/Markov basis. Code comments and small-instance validation are not a general connectivity proof. | `dynamic_augmented_group_counting.tex:188-205`; `augmented/bayesian.py:254-257,345-381,385-519,648-664`; editorial placement in `build_avances_post_sesion_notebook.py:461-462` | **appendix/future** | Appendix for the failure example and exact component fallback; general ergodicity as future work unless a connectivity theorem or exhaustive scoped guarantee is supplied. |
| **Purely dynamic binary results belong to prior group work:** both drafts identify the binary dynamic study as the predecessor; the resolution draft cites Lopez, Marmolejo-Cossío, Tello Ayala, and Parkes, arXiv:2601.22419 (2026). The supplied corpus establishes the intended attribution, but this audit does not independently verify the external paper's theorem statements. | `dynamic_augmented_group_counting.tex:57-60,284-287`; `resolution_three_directions.tex:318-331,351-353`; editorial instruction at `build_avances_post_sesion_notebook.py:463` | **qualified** | Related Work/Positioning. Any dynamic-only theorem or empirical finding must be explicitly attributed rather than presented as a contribution of the unified paper. |

## Source reconciliation and unresolved ambiguities

1. **Reward convention was inconsistent in the source drafts and is resolved in
   the unified manuscript.** The unified theorem uses strict hard clearing, adds
   the final singleton test, and states \(B=k+\log_2G+1\). Its anchor
   \((q,G,k,B)=(0.05,16,2,7)\) gives \(0.806u>0.35u\), and its optimized integer
   coverage is \(2^{B-2}\).

2. **The main empirical table now has a traceable canonical source.** The two raw
   hierarchy CSVs and their summaries reproduce the three reported relative-gain
   estimates with 200/200/40 instances and zero chain violations. The paper must
   define the estimator as the mean of the per-instance ratios; computing a ratio
   from the displayed mean welfare columns gives different percentages. The
   unified table now uses 200/200/40 instances and labels the estimator explicitly.

3. **The source drafts mixed verified special cases with proposed general cases;
   the unified manuscript separates them.** It gives scoped propositions for
   disconnected components, fully observed laminar layers, and the acyclic
   one-bit-separator chain. Partially observed laminar trees and a general
   bounded-treewidth solver are stated as outside the current scope. The chain's
   primal graph is correctly identified as having treewidth 2.

4. **The resolution data are internally reproducible but narrowly scoped.** The
   84.5% number is exactly supported by `resolution_curve.csv`; causal language
   about why caps saturate and universal language about the “only” interior level
   are reduced in the unified manuscript to observations on eight designed
   instances, with the selected instance's \(p\) and \(u\) reported.

5. **Complexity and MCMC claims need proof repair, not rhetorical promotion.** The
   source draft's normalizer-hardness sketch is not included as a theorem in the
   unified manuscript, which leaves both normalizer and marginal complexity open.
   The alternating-path implementation likewise does not prove irreducibility on
   all fibers.

## Explicit check of the four editorial rules

1. **Do not attribute a two-variable comparison to one mechanism — satisfied,
   with the intermediate analytic cell still open.** The manuscript defines
   \(U^{\mathrm{stat,binary}}\), \(U^{\mathrm{dyn,binary}}\), and
   \(U^{\mathrm{dyn,count}}\), calls the theorem a joint separation throughout,
   and states in the introduction and limitations that the middle optimum on the
   homogeneous family is pending. The finite empirical hierarchy reports the
   middle cell but does not substitute for the analytic comparison.

2. **Keep posterior \#P hardness and fiber enumeration out of the main line —
   satisfied.** The abstract and contribution list omit them. The finite
   5-versus-25 illustration, sampler limitations, and open posterior-complexity
   questions appear only in Appendix B.

3. **Attribute purely dynamic binary work to the group's predecessor — satisfied.**
   The introduction assigns the dynamic-binary model and findings to Lopez et al.
   (2026), and the new contributions are the joint separation, structured
   count-posterior algorithms, and count/resolution evidence.

4. **The sendable triad is separation + efficient tractable-regime algorithm +
   empirical evidence — satisfied.** The main text contains the corrected strict-
   clearing separation, scoped component/laminar/chain algorithms, and two traced
   empirical sections. Complexity hardness and fiber sampling remain appendix
   material; RL, certificates, and the broad three-knob map are not competing
   headline contributions.
