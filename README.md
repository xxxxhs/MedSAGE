# MedSAGE

Official implementation and experimental materials for the manuscript:

**Improving Medical Question Answering in Online Health Consultations Through Retrieval-Augmented Semantic Aggregation of Multiple Physician Responses**

MedSAGE is a medical semantic aggregation framework designed for the "one question, multiple answers" setting in online health consultation. It integrates semantic centrality, query relevance, and evidence diversity to construct informative evidence contexts for medical question answering.

This repository provides the MedSAGE implementation, experiment configurations, and finalized question-level materials corresponding to the quantitative analyses reported in the manuscript.

## Method overview

MedSAGE ranks candidate evidence using three complementary signals:

- **Semantic Centrality (C):** structural importance within the semantic similarity graph;
- **Query Relevance (R):** semantic relevance to the target medical question;
- **Diversity (D):** complementary information relative to selected evidence.

The three components are normalized within each candidate set and combined through weighted aggregation:

```text
Score = wC * C_norm + wR * R_norm + wD * D_norm
```

The default MedSAGE configuration is:

```text
C:R:D = 1:1:1
K = 10
tau = 0.7
```

## Experimental materials

Finalized manuscript-related materials are organized under:

```text
reproducibility_results/
├── Table 1/   Main model comparison
├── Table 3/   Physician evaluation
├── Table 4/   Retrieval and aggregation comparison
├── Table 5/   Component ablation
├── Table 6/   Evidence-size sensitivity analysis
├── Table 7/   Similarity-threshold sensitivity analysis
├── Table 8/   C/R/D coefficient sensitivity analysis
└── Table 9/   Medical entity consistency analysis
```

The model backbones used in the manuscript include:

```text
DeepSeek-R1
Qwen3-32B
LLaMA4-17B
HuatuoGPT-o1-7B
```

Table 1 reports the main comparison across Base, RAG, and MedSAGE settings. Tables 4–8 contain the controlled comparison, ablation, and sensitivity analyses. Table 3 contains physician-evaluation materials, and Table 9 contains the medical entity consistency analysis.

The question-level result files provide the generated outputs and corresponding evaluation information used for the reported analyses.

## Repository structure

```text
MedSAGE/
├── configs/                    Experiment configurations
├── docs/                       Supporting documentation
├── examples/                   Example input formats
├── medsage/                    Core MedSAGE implementation
├── scripts/                    Experiment scripts
├── outputs/                    Runtime outputs
├── reproducibility_results/    Finalized manuscript-related results
├── results/                    Earlier result release
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

## Experimental settings

The released configurations cover the main experiments reported in the manuscript, including:

```text
Main comparison:
Base / RAG / MedSAGE

Aggregation comparison:
RAG / RAG+Avg / RAG+MMR / MedSAGE

Component analysis:
C / R / D / C+R / R+D / C+D / C+R+D

Evidence size:
K = 3, 5, 8, 10, 12

Similarity threshold:
tau = 0.5, 0.6, 0.7, 0.8, 0.9
```

Additional C/R/D weighting configurations follow the settings reported in the manuscript.

## Evaluation

The automatic evaluation metrics used in the manuscript include:

```text
BLEU-2
ROUGE-L
METEOR
CIDEr
BERTScore-F1
```

Question-level outputs and metric information are provided in the corresponding experimental result directories.

## Implementation note

The released implementation corresponds to the finalized MedSAGE pipeline used for the reported quantitative experiments. The semantic graph uses binary edges determined by the similarity threshold, centrality is calculated as binary node degree, and centrality, relevance, and diversity are normalized before weighted aggregation.

An earlier supplementary version contained an illustrative intermediate-scoring example originating from a prototype implementation. That example was not used to generate the quantitative results reported in Tables 1 and 4–8. The current implementation and released result materials correspond to the finalized experimental pipeline.

## Usage

Install the project dependencies and configure the required local paths before running the experiment scripts.

```bash
pip install -r requirements.txt
```

The main experiment entry points are available in the `scripts/` directory, with configuration files provided under `configs/`.

## Citation

If you use MedSAGE or the released experimental materials in your research, please cite the corresponding manuscript.

## License

This code is released under the MIT License. Associated datasets and model resources are governed by their respective licenses.
