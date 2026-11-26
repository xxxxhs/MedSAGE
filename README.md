# MedSAGE

Research implementation and experimental materials accompanying the manuscript:

**Improving Medical Question Answering in Online Health Consultations Through Retrieval-Augmented Semantic Aggregation of Multiple Physician Responses**

MedSAGE integrates sentence-level semantic centrality (C), query relevance (R), and diversity (D) through normalized weighted scoring and greedy iterative evidence selection. The default configuration is K = 10, tau = 0.7, C:R:D = 1:1:1.

## Revised manuscript materials — 2026-11-26

The directories below contain the archived results and supplementary materials for the revised manuscript. Earlier releases remain available in the repository and its commit history. The revised table numbering differs from the earlier release.

| Manuscript item | Archived materials |
| --- | --- |
| Table 1 — Retrospective source-linked retrieval | [result/table-1](result/table-1/) |
| Table 2 — Semantic evidence coverage and diversity | [result/table-2](result/table-2/) |
| Table 3 — Full-scale cross-model generation | [result/table-3](result/table-3/) |
| Table 4 — Physician evaluation | [result/table-4](result/table-4/) |
| Table 5 — Controlled baseline comparison | [result/table-5](result/table-5/) |
| Table 6 — Component ablation | [result/table-6](result/table-6/) |
| Table 7 — Evidence selection size | [result/table-7](result/table-7/) |
| Table 8 — Semantic similarity threshold | [result/table-8](result/table-8/) |
| Table 9 — C/R/D aggregation coefficients | [result/table-9](result/table-9/) |
| Figure 4 — Medical question categories | [result/Fig-4](result/Fig-4/) |
| Figure 5 — Physician rating distributions | [result/Fig-5](result/Fig-5/) |

### Supplementary materials

- [Appendix A](Appendix/Appendix_A_Question_IDs/): fixed question identifiers for the 49,176-question, 10,000-question and 200-question evaluations.
- [Appendix D](Appendix/Appendix_D_Medical_Terminology/): terminology dictionary (671 entries).
- [Appendix E](Appendix/Appendix_E_Physician_Evaluation/): item-level ratings and summary.
- [Appendix F](Appendix/Appendix_F_Statistical_Analysis/): statistical comparisons and inter-rater reliability.
- [Appendix G](Appendix/Appendix_G_Retrieval_Robustness/): supplementary retrieval analyses.

Appendices B, C, H and I are described in the accompanying Multimedia Appendix 1. The category-specific source materials are in Figure 4's directory. No separate Appendix I data directory is included in this revision archive.

### Retrieval and generation protocol

The full-scale retrospective retrieval evaluation retains historical evidence associated with the query's own question ID. The archived full-scale generation inputs likewise retain same-ID evidence; they link to the frozen Stage 3 retrieval outputs. The earlier manuscript description of same-ID exclusion for full-scale generation was incorrect.

See the [retrieval manifest](result/table-1/Table1_Retrieval_Manifest.json), [generation-input manifest](result/table-3/Generation_Inputs/Generation_Input_Manifest.json), [generation inputs](result/table-3/Generation_Inputs/) and [knowledge base](result/table-3/Knowledge_Base/). Reference-answer overlap can affect automated similarity scores and may affect comparisons between retrieval strategies. The full-scale results characterize retrospective evidence-augmented generation, not independent generalization to unseen questions. The same-ID policy in the controlled 10,000-question runs requires separate verification. No conclusion from a same-ID-excluded rerun or post-hoc impact analysis is asserted here.

### Access and reproducibility scope

The archived files include generated responses, item-level evaluation results, summaries and selected manifests. The four model names follow the manuscript labels; model identity and historical execution provenance require the corresponding original API/checkpoint and run records. The archive supports inspection of the released materials, but does not establish complete end-to-end reproducibility of every experiment.

Three large generation-input JSONL files are tracked with Git LFS. To retrieve the archive, install Git LFS and run:

```bash
git clone https://github.com/xxxxhs/MedSAGE.git
cd MedSAGE
git lfs pull
git lfs fsck
```

The [Revision_Patch](Revision_Patch/) directory contains an additive reference implementation and selected original evaluation scripts. See its [README](Revision_Patch/README.md) and [CHANGELOG](Revision_Patch/CHANGELOG.md) for scope and limitations. Historical configurations and scripts may use earlier numbering or different protocols. They should not be treated as verified records of every experiment in the revised manuscript.

## Earlier releases

- [reproducibility_results/](reproducibility_results/) — earlier manuscript-related results.
- [results/](results/) — earlier result release.
- [Previous README](https://github.com/xxxxhs/MedSAGE/blob/bf36e798b6880222a71e29590a5301e22c3af4f6/README.md) — historical documentation and table numbering.

Original code, configurations, data files and manifests are retained. Consult the revised manuscript and Multimedia Appendix 1 for the current evaluation definitions.

## Citation and license

Please cite the accompanying manuscript when using MedSAGE. Code is released under the MIT License. Dataset and model resources are subject to their respective licenses.
