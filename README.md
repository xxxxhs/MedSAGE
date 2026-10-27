# MedSAGE

This repository contains the complete experimental implementation and finalized reproducibility materials for the manuscript:

**Improving Medical Question Answering in Online Health Consultations Through Retrieval-Augmented Semantic Aggregation of Multiple Physician Responses**

MedSAGE is a medical semantic aggregation module designed for the "one question, multiple answers" setting in online health consultation. This repository provides the code workflow for preprocessing eligible multi-answer data, dense retrieval, evidence construction, LLM generation, automatic evaluation, physician evaluation, ablation experiments, hyperparameter sensitivity analysis, reference-supported medical entity consistency analysis, and statistical testing.

The repository is intended to support reproduction and verification of the **complete experimental workflow and finalized reported results**.

## Finalized reproducibility materials

The finalized files corresponding to the current manuscript are provided in:

```text
reproducibility_results/
├── Table 1/   Complete 49,176-question generated outputs and per-question metrics
├── Table 3/   Blinded physician-evaluation outputs and item-level ratings
├── Table 4/   Controlled 2,000-question baseline comparison and per-question metrics
├── Table 5/   Component-level ablation outputs and per-question metrics
├── Table 6/   Evidence-size sensitivity outputs and per-question metrics
├── Table 7/   Similarity-threshold sensitivity outputs and per-question metrics
├── Table 8/   C/R/D coefficient sensitivity outputs and per-question metrics
└── Table 9/   Medical-entity dictionary, category statistics, and per-question audit
```

The final model names used in the manuscript and released files are:

```text
DeepSeek-R1
Qwen3-32B
LLaMA4-17B
HuatuoGPT-o1-7B
```

For Table 1, each model is evaluated under `Base`, `RAG`, and `MedSAGE` settings on the complete 49,176-question multi-answer subset.

For Tables 4–8, the controlled analyses use the same fixed 2,000-question evaluation set with HuatuoGPT-o1-7B. Table 3 contains the blinded physician-evaluation materials, and Table 9 contains the reference-supported medical entity consistency audit.

> **Important:** The `results/` directory is retained only as a historical/superseded release for version history. The files in `reproducibility_results/` are the finalized materials corresponding to the current manuscript and should be used for verification.

## What is not included

This repository does **not** redistribute the complete original cMedQA2 source dataset, large language model weights, sentence embedding model weights, FAISS index files, API keys, or private configuration files. Users should obtain datasets and model weights from their original sources.

The generated outputs and per-question result files required to verify the manuscript tables are included under `reproducibility_results/`.

## Repository structure

```text
MedSAGE/
├── configs/                    Configuration files and example paths
├── docs/                       Data, model, and reproducibility documentation
├── examples/                   Small example files for format checking
├── medsage/                    Core implementation
├── scripts/                    End-to-end experiment scripts
├── outputs/                    Runtime outputs, ignored by Git except .gitkeep
├── reproducibility_results/    Finalized results for Tables 1 and 3–9
├── results/                    Superseded historical result release
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
└── CITATION.cff
```

## Installation

```bash
conda create -n medsage python=3.10
conda activate medsage
pip install -r requirements.txt
```

## Quick start

1. Download cMedQA2 and place the files as:

```text
data/cmedqa2/question.csv
data/cmedqa2/answer.csv
```

2. Copy the path template and modify local data/model paths:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

3. Run the full pipeline for one model:

```bash
python scripts/run_full_pipeline.py --config configs/default.yaml --paths configs/paths.yaml --model huatuogpt7b
```

Use `--dry-run` to generate prompts/evidence without running LLM inference.

## Main experiment commands

```bash
python scripts/01_prepare_cmedqa2.py --config configs/default.yaml --paths configs/paths.yaml
python scripts/02_build_faiss_index.py --config configs/default.yaml --paths configs/paths.yaml
python scripts/03_retrieve_evidence.py --config configs/default.yaml --paths configs/paths.yaml
python scripts/04_run_generation.py --config configs/default.yaml --paths configs/paths.yaml --scheme medsage --model huatuogpt7b
python scripts/05_evaluate_main_results.py --config configs/default.yaml --paths configs/paths.yaml
python scripts/08_run_ablation_study.py --config configs/default.yaml --paths configs/paths.yaml --model huatuogpt7b
python scripts/09_run_hyperparameter_sensitivity.py --config configs/default.yaml --paths configs/paths.yaml --model huatuogpt7b
python scripts/10_entity_consistency_analysis.py --config configs/default.yaml --paths configs/paths.yaml
python scripts/11_statistical_tests.py --config configs/default.yaml --paths configs/paths.yaml --model huatuogpt7b
```

## Data note

The complete original cMedQA2 dataset should be obtained from the original public source. The dataset itself is not redistributed here. The released reproducibility files retain the identifiers needed to verify the reported analyses.

## Model note

The LLM checkpoints and m3e-small embedding model are not included. Users should download the models from the corresponding official sources and configure their local paths in `configs/paths.yaml`.

## License

This code is released under the MIT License. Data and model weights are governed by their original licenses.
