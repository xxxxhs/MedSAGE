# MedSAGE — Revision Patch

> **Revision patch (2026-11-26).** This directory is an additive release for the revised manuscript. The repository's previously released files outside this directory are preserved. This is an overlay, not a standalone project. See [CHANGELOG.md](CHANGELOG.md) for provenance, application instructions, and limitations. Paths in the instructions below refer to the repository root after applying the overlay to a separate working copy; links to experimental materials are not a claim that every dependency is available here.

Research implementation and experimental materials accompanying the manuscript:

**Improving Medical Question Answering in Online Health Consultations Through Retrieval-Augmented Semantic Aggregation of Multiple Physician Responses**

MedSAGE organizes evidence from historical physician responses using sentence-level semantic centrality, query relevance, and diversity. It combines normalized component scores with greedy iterative evidence selection before retrieval-augmented answer generation.

## Method

- Centrality (C): binary node degree in a thresholded sentence-similarity graph.
- Relevance (R): cosine similarity between a candidate sentence and the query.
- Diversity (D): one minus the maximum similarity to previously selected sentences; initialized to one.
- Component normalization: min-max scaling, with constant components assigned one.
- Default configuration: K = 10, tau = 0.7, C:R:D = 1:1:1.

The normalization pool and evidence construction depend on the experimental protocol. The modular implementation under `medsage/` is a reference implementation and should not be assumed to reproduce every historical experiment exactly.

## Current manuscript evaluation

The raw cMedQA2 files processed in the study contained 120,000 questions and 226,266 historical responses. The multi-answer subset contained 49,176 questions and 155,442 responses.

| Manuscript item | Evaluation |
| --- | --- |
| Table 1 | Retrospective source-linked retrieval: nDCG@10, P@10, R@10, MAP@10, MRR@10, EAC@10 |
| Table 2 | Automatic semantic evidence coverage and redundancy |
| Table 3 | Four-model comparison: Base, RAG, RAG+MMR, MedSAGE |
| Table 4 | Two physicians' ratings of 3,200 responses to 200 questions |
| Table 5 | Controlled comparison: Base, RAG, RAG+Avg, RAG+MMR, MedSAGE |
| Table 6 | Centrality, relevance, and diversity ablation |
| Table 7 | Evidence selection size K |
| Table 8 | Similarity threshold tau |
| Table 9 | C/R/D coefficient sensitivity |
| Figure 4 | Terminology-defined medical question categories |
| Figure 5 | Physician rating distributions |

The manuscript reports DeepSeek-R1, Qwen3-32B, LLaMA4-17B, and HuatuoGPT-o1-7B as the four generation models. The manuscript reports HuatuoGPT-o1-7B for the controlled evaluation and representative Figures 4 and 5. These are manuscript-reported labels; exact model checkpoints/API identifiers and execution provenance must be established from the corresponding original run records, not from configuration aliases alone.

The controlled analysis uses a fixed 10,000-question subset (reported sampling seed 202612); physician evaluation uses 200 questions (reported sampling seed 202611). Each physician assessment condition contains 200 responses. Fixed question IDs, rather than regeneration of the sample from a seed alone, are needed for exact reproduction.

## Retrospective retrieval evaluation (Tables 1–2)

These experiments evaluate historical evidence recovery, not independently annotated clinical relevance. They intentionally retain evidence associated with the query's own question identifier.

Original evaluation scripts are archived under:

```text
scripts/retrieval_evaluation/
  04_evaluate_source_linked_retrieval.py
  05_evaluate_semantic_coverage.py
```

Both scripts evaluate previously saved Top-10 retrieval outputs for 49,176 questions. Stage 4 calculates source-linked metrics and paired bootstrap estimates. Stage 5 calculates automatic semantic facet coverage, cross-source coverage, multi-answer support, consensus-aspect hits, pooled alpha-nDCG, and near-duplicate pair fraction.

The documented retrospective settings include a 50-sentence MMR candidate pool, MMR coefficient 0.5, semantic clustering and matching thresholds 0.82, redundancy threshold 0.85, and alpha = 0.5. The Stage 4 and Stage 5 scripts retain their original evaluation logic and environment-variable configuration.

To run them, set `MEDSAGE_STAGE4_BASE` and `MEDSAGE_STAGE5_BASE` to the experiment root containing their required upstream files. These include the frozen Stage 3 retrieval outputs and manifest, the expert knowledge-base SQLite database for Stage 4, and the question records, sentence position mapping, and sentence embeddings for Stage 5. Consult the scripts for exact paths and validation requirements.

These scripts do not build the original indexes or generate new retrieval outputs. Full execution requires the corresponding upstream artifacts; the availability of all such dependencies is not asserted here.

## Generation and evaluation settings

The documented generation settings (Multimedia Appendix 1, Tables S3 and S6) are temperature 0.7, top-p 0.9, maximum output 750 tokens, and explicit thinking disabled where supported. Chinese responses are requested to contain approximately 450 characters (400–500); this is a prompt instruction, not an acceptance filter.

The updated prompt implementation is in `medsage/prompting.py`, with templates in `examples/sample_prompt.txt`. RAG+Avg uses the mean candidate-sentence embedding, excluding the query embedding.

The manuscript specifies same-question-ID exclusion for downstream generation, unlike retrospective retrieval. The reference retrieval module defaults to exclusion; this does not independently establish that every historical generation input was filtered. Original retrieval and generation manifests must be consulted to verify the actual protocol.

Automatic generation metrics: BLEU-2, ROUGE-L, BERTScore-F1, METEOR-lite, and CIDEr-lite. The first four use metric-specific best-match reference scoring; CIDEr-lite uses an averaged multi-reference representation. The lite metrics must not be interpreted as standard METEOR or CIDEr.

Medical question stratification uses the 671-entry terminology dictionary described in Appendix D. The small YAML lexicon included in the reference implementation is an example, not the formal study dictionary. Physician assessment and statistical protocols are described in Appendices E and F.

## Repository layout

```text
configs/                      Configuration and example paths
medsage/                      Modular reference implementation
scripts/                      Experiment and reference workflow scripts
scripts/retrieval_evaluation/ Original Stage 4 and Stage 5 evaluators
examples/                     Example inputs and prompts
docs/                         Supporting documentation (may describe earlier workflows)
outputs/                      Runtime outputs
results/                      Result materials
reproducibility_results/      Earlier reproducibility materials
```

Historical scripts and documents may retain earlier experimental numbering or exploratory metrics (including entity consistency and Distinct-n). These should not be confused with the current manuscript's Tables 1–9. See the current manuscript and Multimedia Appendix 1 for authoritative evaluation definitions and the current table numbering.

## Usage and scope

Install dependencies with `pip install -r requirements.txt`, then configure local paths using `configs/paths.example.yaml` as a template for `configs/paths.yaml`. Obtain the source dataset and model resources from their original providers, subject to their licenses.

The modular generation workflow is a reference workflow. It currently implements local Hugging Face inference; it does not provide the original API inference services, full protocol-specific retrieval construction, all fixed-sample workflows, or all statistical and physician-rating analyses. The documented protocol settings in the configuration are not a replacement for the original execution records. Running `scripts/run_full_pipeline.py` is not a guarantee of end-to-end reproduction of the revised manuscript. In particular, the legacy entry point does not enable BERTScore by default.

The current manuscript and Multimedia Appendix 1 describe dataset sampling (A), implementation and prompts (B), retrieval metrics (C), terminology dictionary (D), physician evaluation (E), statistical analyses (F), robustness analyses (G), and category-specific results (H).

## Citation and license

Please cite the accompanying manuscript when using MedSAGE. Code is distributed under the repository license; dataset and model resources remain subject to their original licenses.
