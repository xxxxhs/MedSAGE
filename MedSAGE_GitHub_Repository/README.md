# MedSAGE

This repository contains the complete experimental implementation for the manuscript:

**Improving Medical Question Answering in Online Health Consultations Through Retrieval-Augmented Semantic Aggregation of Multiple Physician Responses**

MedSAGE is a medical semantic aggregation module designed for the "one question, multiple answers" setting in online health consultation. This repository provides the complete code workflow for preprocessing all eligible multi-answer data, dense retrieval, evidence construction, LLM generation, main evaluation, terminology-category evaluation, ablation experiments, hyperparameter sensitivity analysis, reference-supported medical entity consistency analysis, and statistical testing.

This repository is intended to reproduce the **complete experimental workflow** reported in the manuscript, rather than only the additional analyses added during revision.

## What is not included

This repository does **not** include the full cMedQA2 dataset, large language model weights, sentence embedding model weights, generated full-output files, FAISS index files, API keys, or private configuration files. Users should obtain the dataset and model weights from their original sources.

## Repository structure

```text
MedSAGE/
├── configs/                 Configuration files and example paths
├── docs/                    Data, model, and reproducibility documentation
├── examples/                Small example files for format checking
├── medsage/                 Core implementation
├── scripts/                 End-to-end experiment scripts
├── outputs/                 Runtime outputs, ignored by Git except .gitkeep
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

The complete cMedQA2 dataset should be obtained from the original public source. The dataset is not redistributed here.

## Model note

The LLM checkpoints and m3e-small embedding model are not included. Users should download the models from the corresponding official sources and configure their local paths in `configs/paths.yaml`.

## License

This code is released under the MIT License. Data and model weights are governed by their original licenses.
