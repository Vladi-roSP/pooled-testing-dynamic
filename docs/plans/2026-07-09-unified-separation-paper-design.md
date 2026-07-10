# Unified Separation-First Paper Design

## Objective

Merge `augmented/paper/dynamic_augmented_group_counting.tex` and
`augmented/paper/resolution_three_directions.tex` into one English manuscript
with a single, defensible publication story. The main line must contain three
elements: a separation example, an efficient algorithm for tractable regimes,
and empirical evidence.

## Editorial thesis

The paper asks when count-valued outcomes create welfare that binary group
tests cannot recover. It opens with a homogeneous toy instance in which the
best static binary design has value $B u q$, while a dynamic count-valued
strategy searches exponentially more people and guarantees a healthy
individual whenever one exists in the covered population. This example
establishes the separation without claiming that the paper has already
isolated the contribution of adaptivity from the contribution of resolution.

The intermediate comparison against the optimal dynamic binary policy remains
an explicitly pending cell. Existing results about purely dynamic binary group
testing are attributed to the group's previous work.

## Manuscript architecture

1. **Introduction and separation preview.** Begin with a small numerical toy
   example before notation. State the publication claim and the unresolved
   dynamic-binary comparison.
2. **Model and comparison classes.** Define static binary, dynamic binary, and
   dynamic augmented policies without implying that two simultaneous changes
   identify either mechanism separately.
3. **Separation result.** Present the homogeneous family, prove the static
   optimum, construct the dynamic count strategy, and state the parameter
   window where the lower bound separates.
4. **Efficient algorithms in tractable regimes.** Cover disjoint, laminar,
   chain, bounded-treewidth, and few-test structure at the level justified by
   current results. Separate proved algorithms from proposed extensions.
5. **Empirical evidence.** Retain the strongest existing count-vs-binary and
   resolution-curve experiments. Treat resolution as supporting evidence, not
   as a competing central thesis.
6. **Discussion and limitations.** Name the missing optimal dynamic-binary
   comparison, the idealized exact-count channel, and scale limits.
7. **Appendix or future work.** Move #P-hard posterior inference, exact fiber
   enumeration, Gibbs ergodicity details, and broader certificate directions
   out of the main narrative.

## Use of the Francisco style skill

The manuscript will adapt the principles, not imitate Spanish conversational
phrases. Intuition precedes formalism; a concrete toy population precedes the
general theorem; the explanation reduces mechanisms to counting; and each
section reconnects its result to the publication claim. Formal statements,
proofs, citations, and empirical reporting remain concise academic English.

## Source policy

The two original TeX files remain unchanged as source snapshots. The merged
paper is written to a new stable TeX file. Claims and numbers are retained only
when they can be traced to the source papers, the post-session notebook, or
generated project data. Unsupported or internally inconsistent claims are
removed or marked as pending.

## Verification

The completed manuscript must:

- compile without LaTeX errors;
- render without clipping, overlap, broken glyphs, or malformed references;
- satisfy all four post-session publication rules;
- distinguish static binary, dynamic binary, and dynamic augmented regimes;
- place #P hardness and fiber enumeration outside the main line;
- attribute purely dynamic binary work to prior group work;
- contain the separation, tractable algorithmic regimes, and empirical evidence;
- avoid invented citations, results, and numerical claims.
