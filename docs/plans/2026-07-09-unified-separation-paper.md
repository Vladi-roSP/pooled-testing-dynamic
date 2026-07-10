# Unified Separation-First Paper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce one English LaTeX manuscript that merges the two existing AI-assisted drafts around a separation-first publication story and satisfies the four post-session editorial rules.

**Architecture:** Preserve both source drafts and create a new manuscript. Start from a claim audit so every theorem, algorithmic statement, and number maps to a source or generated dataset. Organize the main text as separation, tractable algorithms, and empirical evidence; keep #P hardness, fiber enumeration, Gibbs details, and certificate extensions in appendices or future work.

**Tech Stack:** LaTeX, project CSV data, Jupyter notebook sources, `rg`, `jq`, `pdflatex`, Poppler (`pdfinfo`, `pdftoppm`), and read-only consistency scripts.

---

### Task 1: Audit both drafts against the post-session notebook

**Files:**
- Read: `augmented/paper/dynamic_augmented_group_counting.tex`
- Read: `augmented/paper/resolution_three_directions.tex`
- Read: `augmented/notebooks/avances_post_sesion.ipynb`
- Read: `augmented/notebooks/build_avances_post_sesion_notebook.py`
- Create: `augmented/paper/unified_claim_audit.md`

**Step 1: Extract both outlines**

Run: `rg -n '^\\(title|begin\{abstract\}|section|subsection|paragraph|begin\{theorem\}|begin\{lemma\}|begin\{proposition\}|begin\{corollary\}|begin\{example\})' augmented/paper/dynamic_augmented_group_counting.tex augmented/paper/resolution_three_directions.tex`

Expected: both paper structures, including inference hardness, resolution, structure, and benchmarking.

**Step 2: Extract the editorial constraints**

Run: `jq -r '.cells[] | select(.cell_type=="markdown") | .source | join("")' augmented/notebooks/avances_post_sesion.ipynb | rg -n -C 5 'El ejemplo de separación|Encuadre para publicar|regímenes tratables'`

Expected: the separation formulas, tractable-regime discussion, and all four publication rules.

**Step 3: Write the claim audit**

Create a table with `Claim`, `Source`, `Evidence status`, and `Destination`. Classify the static optimum, dynamic-count lower bound, missing dynamic-binary comparison, refinement monotonicity, every tractable regime, empirical percentages, resolution values, #P hardness, fiber enumeration, Gibbs ergodicity, and prior dynamic-binary work. Use only `verified`, `qualified`, `pending`, or `appendix/future` as statuses.

**Step 4: Verify coverage**

Run: `rg -n 'dynamic-binary|#P|fiber|prior group work|separation|tractable|empirical' augmented/paper/unified_claim_audit.md`

Expected: every editorial rule has an explicit audit entry.

**Step 5: Commit**

Run: `git add augmented/paper/unified_claim_audit.md` then `git commit -m "docs(paper): audit claims for unified manuscript"`.

### Task 2: Scaffold the unified manuscript and write the separation

**Files:**
- Create: `augmented/paper/unified_dynamic_augmented_group_counting.tex`
- Reference: `augmented/paper/separacion_regimen_francisco.md`
- Reference: `augmented/notebooks/avances_post_sesion.ipynb`

**Step 1: Create the section shell**

Use this order: Abstract; Introduction; Model and Comparison Classes; A Separation for Dynamic Count-Valued Testing; Efficient Computation in Structured Regimes; Empirical Evidence; Discussion and Limitations; Conclusion; Appendix A, Resolution Monotonicity and Additional Experiments; Appendix B, Posterior Complexity and Fiber Computation.

**Step 2: Open with a toy example**

Introduce a concrete population, budget, pool size, and healthy probability before general notation. Regenerate its arithmetic from the notebook formulas.

**Step 3: Define the three comparison classes**

Define static binary, dynamic binary, and dynamic count-valued policies separately. State that the proved example compares the first and third, changing both adaptivity and outcome resolution.

**Step 4: Prove the static benchmark**

Show that a group of size $g$ yields $u g q^g$ and that $gq^{g-1}\le 1$ for $q<1/2$. Include the overlap argument only at the strength justified by the notebook.

**Step 5: State the dynamic-count lower bound**

Give the construction with $B=k+\lceil\log_2G\rceil$, or include the extra certification test. State the accreditation convention explicitly and derive $U^{\mathrm{dyn,count}} \ge u(1-(1-q)^{kG})$.

**Step 6: Preserve the pending comparison**

State that the optimal dynamic-binary comparison is needed to attribute the gain between adaptivity and resolution and remains pending.

**Step 7: Compile and check terminology**

Run: `pdflatex -interaction=nonstopmode -halt-on-error -output-directory augmented/paper augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Run: `rg -n 'static binary|dynamic binary|dynamic count|pending|remains open|prior work' augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Expected: exit code 0 and explicit mention of all three regimes and the missing cell.

**Step 8: Commit**

Run: `git add augmented/paper/unified_dynamic_augmented_group_counting.tex` then `git commit -m "docs(paper): add unified separation-first manuscript"`.

### Task 3: Integrate tractable computation without overstating results

**Files:**
- Modify: `augmented/paper/unified_dynamic_augmented_group_counting.tex`
- Reference: `augmented/notebooks/build_avances_post_sesion_notebook.py`
- Reference: `augmented/paper/escalabilidad_e_inferencia.md`
- Reference: `augmented/bayesian.py`
- Reference: `augmented/solver.py`

**Step 1: Separate implemented results from proposed extensions**

Use the audit to determine which regimes have executable support. Present disjoint components and few-test enumeration as implemented only if the code supports that wording. Present laminar, chain, and bounded-treewidth methods at the exact verified level.

**Step 2: Explain the common mechanism**

Describe how structural separators limit posterior state and avoid global enumeration over all $2^N$ profiles. Keep #P hardness out of this section except for one pointer to Appendix B.

**Step 3: Connect inference to decisions**

Explain how the posterior representation feeds greedy or exact dynamic decisions. Do not imply a scalable exact optimizer if the evidence establishes only scalable inference.

**Step 4: Audit strong language**

Run: `rg -n 'we prove|polynomial|exact|implemented|future|proposed|#P' augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Expected: every strong algorithmic word is supported by `unified_claim_audit.md`; #P is only a forward pointer in the main line.

**Step 5: Compile and commit**

Run the same `pdflatex` command as Task 2, then add the TeX file and commit with `docs(paper): integrate tractable computational regimes`.

### Task 4: Consolidate verified empirical evidence

**Files:**
- Modify: `augmented/paper/unified_dynamic_augmented_group_counting.tex`
- Modify: `augmented/paper/unified_claim_audit.md`
- Read: `augmented/data/resolution_curve.csv`
- Read: `augmented/data/nick_style_augmented_small_summary.csv`
- Read: `augmented/data/results_N4_B2_G2.csv`
- Read: `augmented/data/results_N5_B2_G3.csv`

**Step 1: Recompute every reported number**

Use read-only aggregation. Record the source file and column behind every percentage and table entry in the audit.

**Step 2: Write one empirical story**

Lead with direct dynamic-count versus dynamic-binary results. Follow with the resolution curve as a mechanism experiment. Do not make horizon, structure, and resolution simultaneous headline contributions.

**Step 3: Put limitations beside results**

Report population sizes, budgets, pool limits, sample counts, and whether each value is optimal or heuristic.

**Step 4: Check traceability**

Run: `rg -n '\\%|[0-9]+\.[0-9]+' augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Expected: every empirical number maps to a row in the audit.

**Step 5: Compile and commit**

Compile, add the TeX file and audit, then commit with `docs(paper): consolidate verified empirical evidence`.

### Task 5: Relegate secondary theory and apply the adapted voice

**Files:**
- Modify: `augmented/paper/unified_dynamic_augmented_group_counting.tex`
- Read: `/Users/hectorbecerrilvillamil/.claude/skills/estilo-francisco/SKILL.md`
- Read: `/Users/hectorbecerrilvillamil/.claude/skills/estilo-redaccion/SKILL.md`

**Step 1: Write Appendix A**

Move refinement monotonicity and additional resolution results here. Preserve the deterministic-garbling proof if it remains useful.

**Step 2: Write Appendix B**

Move #P-hard inference, exact fiber enumeration, and Gibbs ergodicity details here. If a reduction or proof has not been independently verified, label it as a proof sketch or future-work claim rather than a theorem.

**Step 3: Attribute prior work**

Refer explicitly to the group's prior dynamic-binary work in positioning. Do not list it among the manuscript's new contributions.

**Step 4: Apply the adapted Francisco voice**

Put intuition and a toy example before formalism, reduce mechanisms to counting comparisons, and reconnect each section to the publication claim. Translate the principles into natural English; do not insert Spanish signature phrases or conversational filler. Keep definitions, proofs, tables, and limitations concise and academic.

**Step 5: Remove AI artifacts and corrupted terminology**

Run: `rg -n -i 'clearancey|latentstates|todo|tbd|obvious|clearly|trivially|in summary|the interesting thing|game-changing|novel' augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Expected: no corrupted terms or placeholders; any remaining strong qualifier is justified.

**Step 6: Check placement**

Run: `rg -n '\\section|#P|fiber|Gibbs|dynamic binary' augmented/paper/unified_dynamic_augmented_group_counting.tex`.

Expected: #P, fiber, and Gibbs details live in appendices or future work; dynamic binary work is attributed.

**Step 7: Compile and commit**

Compile, add the TeX file, and commit with `style(paper): apply separation-first academic voice`.

### Task 6: Verify compilation, rendering, and all four editorial rules

**Files:**
- Verify: `augmented/paper/unified_dynamic_augmented_group_counting.tex`
- Verify: `augmented/paper/unified_dynamic_augmented_group_counting.pdf`
- Modify: `augmented/paper/unified_claim_audit.md`
- Create temporarily: `tmp/pdfs/unified-paper-*.png`

**Step 1: Compile twice**

Run the Task 2 `pdflatex` command twice.

Expected: both runs exit 0 and cross-references settle.

**Step 2: Inspect the log**

Run: `rg -n 'LaTeX Warning|Overfull|Underfull|Undefined|multiply defined|Error' augmented/paper/unified_dynamic_augmented_group_counting.log`.

Expected: no errors, undefined references, or material overfull boxes.

**Step 3: Render every page**

Run: `mkdir -p tmp/pdfs`.

Run: `pdftoppm -png -r 144 augmented/paper/unified_dynamic_augmented_group_counting.pdf tmp/pdfs/unified-paper`.

Run: `pdfinfo augmented/paper/unified_dynamic_augmented_group_counting.pdf`.

Expected: one PNG per page and valid PDF metadata.

**Step 4: Visually inspect all pages**

Check margins, equations, tables, page breaks, headers, footers, references, and glyphs. Fix and repeat until no visual defects remain.

**Step 5: Record compliance**

Confirm in the audit that: the joint comparison is not attributed to one mechanism; #P and fiber work are outside the main line; purely dynamic binary testing is prior group work; and the main triad is separation, tractable algorithms, and empirical evidence.

**Step 6: Final repository checks**

Run: `git diff --check`.

Run: `git status --short`.

Expected: no whitespace errors and no unrelated files staged.

**Step 7: Commit verified artifacts**

Add only the unified TeX, PDF, and audit. Commit with `docs(paper): finalize and verify unified manuscript`.
