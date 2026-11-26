# Revision patch — 2026-11-26

## Scope and provenance

This directory supplements the existing release without replacing or deleting its code, documentation, or experimental results.

- Original repository baseline: [25a3f26355c4ffcccee9ec37e131203223501425](https://github.com/xxxxhs/MedSAGE/commit/25a3f26355c4ffcccee9ec37e131203223501425).
- Source of the revised overlay: [f7eefc635e6a7b031b32897d791e5269adbe0568](https://github.com/xxxxhs/MedSAGE/commit/f7eefc635e6a7b031b32897d791e5269adbe0568) on the manuscript-revision branch.
- The two retrieval evaluation scripts are archived from the supplied Stage 4 and Stage 5 experiment scripts, with filenames standardized and a final newline. Their evaluation formulas and input validation have not been changed.
- The remaining Python/YAML/prompt files are revisions of the reference implementation, not a claim that those revised files were the exact historical run code.

## Changes

| File (relative to this directory) | Change |
| --- | --- |
| scripts/retrieval_evaluation/04_evaluate_source_linked_retrieval.py | Add Stage 4 source-linked evaluation (Table 1). |
| scripts/retrieval_evaluation/05_evaluate_semantic_coverage.py | Add Stage 5 automatic semantic coverage and redundancy evaluation (Table 2). |
| medsage/aggregation.py | RAG+Avg centroid excludes the query embedding and averages candidate sentence embeddings only. |
| configs/default.yaml | Set maximum new tokens to 750 and document protocol-specific settings. |
| medsage/generation.py | Set the default generation token limit to 750. |
| scripts/04_run_generation.py | Set the fallback generation token limit to 750. |
| scripts/08_run_ablation_study.py | Set the fallback generation token limit to 750. |
| medsage/prompting.py | Use the common Chinese system prompt and retrieval-free/evidence-augmented templates documented in Table S6. |
| examples/sample_prompt.txt | Update the prompt example. |
| README.md | Describe the revised manuscript's experimental structure and reproduction scope. |

## Applying the overlay locally

For inspection, no copying or installation is necessary. For running the revised reference implementation, create a separate local working copy of the repository at the original baseline and copy the overlay files into their corresponding relative paths. Do not overwrite the public original release.

For example, in a shell with Git installed:

```sh
git clone https://github.com/xxxxhs/MedSAGE.git MedSAGE-patch-work
cd MedSAGE-patch-work
git switch -c local-revision 25a3f26355c4ffcccee9ec37e131203223501425
git checkout main -- Revision_Patch
```

Then copy the contents of Revision_Patch/configs, Revision_Patch/medsage, Revision_Patch/examples and Revision_Patch/scripts into the matching repository-root directories, preserving relative paths. The patch README and changelog are documentation and do not need to replace the original root README. Install dependencies and configure paths as described in the patch README. Keep changes in the local working copy only.

The two retrospective evaluation scripts can also be inspected or run separately given the necessary upstream artifacts and configured environment variables. They do not reconstruct the original retrieval outputs or indexes.

## Limitations and verification status

This release does not modify results, statistical analyses, medical terminology classification code, physician-rating code, or other historical documentation. The legacy reference workflow remains incomplete for the revised manuscript; see README.md. The MMR value of 0.5 is specific to the retrospective retrieval protocol; the legacy reference generation default of 0.7 is retained pending verification of the downstream settings. Documented sampling and thinking settings are not automatically enforced by all legacy entry points. Exact historical model identifiers, generation prompts and evidence exclusion behavior require original execution records. The overlay does not resolve or retrospectively validate those issues. No experiments have been rerun as part of this patch, and no updated results are claimed.
