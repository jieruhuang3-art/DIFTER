# DIFTER

**Disentangle, Align, and Intervene for Source-Only Cross-Environment Encrypted Traffic Classification**

PyTorch implementation of DIFTER for source-only cross-environment encrypted traffic classification.

## Overview

Encrypted traffic distributions can change across capture times, devices, and application contexts, shifting packet timing, packet-size patterns, and flow behavior. DIFTER learns transferable traffic representations through Class-Conditional Invariant Factorization (CCIF), Cross-Environment Class-Conditional Contrast (CECC), and Compositional Environment Intervention (CEI). Target samples are not used for representation learning or model selection.

The data path is: **Traffic Flow -> Traffic Windows -> HPTF Encoder -> K <= 16 Window Representations -> Masked Mean Aggregation -> CCIF -> CECC / CEI during training -> Stable Classifier**. Here, `K <= 16` refers to traffic windows per flow, not packets.

<p align="center">
  <img src="assets/difter_framework.png" width="100%">
</p>

<p align="center">
  <b>Overview of the DIFTER framework.</b>
</p>

## Highlights

- **CCIF** disentangles a flow representation into a class-stable component and factor-specific environmental components.
- **CECC** aligns same-class stable representations across different source environments.
- **CEI** recombines source-observed environmental factors from same-class donors to regularize unseen factor compositions.

At inference time, DIFTER retains only the HPTF encoder, masked window aggregation, stable projector, and classifier.

## Datasets

| Dataset | Classes | Distribution shift | Source -> Target |
| --- | ---: | --- | --- |
| APP53-Time | 27 | Temporal | Jun. 20--24 -> Jul. 19--23 |
| MIRAGE-2019 | 20 | Device | Device A+B -> Device C |
| MIRAGE-COVID | 9 | Device/activity composition | 10 observed compositions -> held-out composition |

## Main Results

Macro-F1 is reported for source-test and held-out target environments. The target shifts are Jun. 20--24 to Jul. 19--23 for APP53-Time, Device A+B to Device C for MIRAGE-2019, and 10 observed compositions to a held-out composition for MIRAGE-COVID.

| Model | APP53 Source | APP53 Target | MIRAGE-2019 Source | MIRAGE-2019 Target | MIRAGE-COVID Source | MIRAGE-COVID Target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.4174 | 0.2668 | 0.7161 | 0.6672 | 0.7047 | 0.6400 |
| 1D-CNN | 0.1003 | 0.0745 | 0.6262 | 0.5952 | 0.6556 | 0.6082 |
| BiLSTM | 0.1084 | 0.0827 | 0.6594 | 0.6028 | 0.6401 | 0.5956 |
| Vanilla Transformer | 0.0825 | 0.0648 | 0.6484 | 0.5957 | 0.6322 | 0.5895 |
| ET-BERT | 0.0233 | 0.0236 | 0.7601 | 0.7088 | 0.7129 | 0.6496 |
| TrafficFormer | 0.2071 | 0.1194 | 0.7462 | 0.6925 | 0.7124 | 0.6538 |
| HPTF | 0.4845 | 0.2725 | 0.6435 | 0.6077 | 0.6429 | 0.6072 |
| **DIFTER** | **0.5045** | **0.2839** | **0.7163** | **0.7127** | **0.6618** | **0.6616** |

### Same-backbone DG comparison

All methods use the same HPTF backbone, traffic representation, source split, and source-validation model-selection protocol.

| Method | Source F1 | Mean Target F1 | Worst Target F1 |
| --- | ---: | ---: | ---: |
| HPTF-ERM | 0.6429 | 0.6244 | 0.6072 |
| HPTF-CORAL | 0.6454 | 0.6310 | 0.6028 |
| HPTF-GroupDRO | 0.6304 | 0.6068 | 0.5891 |
| HPTF-VREx | 0.5784 | 0.5614 | 0.5442 |
| HPTF-Fishr | 0.6314 | 0.6229 | 0.6068 |
| **DIFTER** | **0.6618** | **0.6428** | **0.6239** |

### Ablation study

| Variant | CCIF | CECC | CEI | Macro-F1 | Delta vs. Base |
| --- | :---: | :---: | :---: | ---: | ---: |
| HPTF Base | - | - | - | 0.4908 | - |
| + CCIF | ✓ | - | - | 0.5133 | +0.0225 |
| + CCIF + CECC | ✓ | ✓ | - | 0.5322 | +0.0414 |
| + CCIF + CEI | ✓ | - | ✓ | 0.5359 | +0.0451 |
| **DIFTER** | ✓ | ✓ | ✓ | **0.5642** | **+0.0734** |

CCIF establishes the factorized representation space, while CECC and CEI provide complementary improvements.

## Method

### CCIF

CCIF separates each flow representation into a class-stable component and factor-specific environmental components. The classifier operates on the stable component.

### CECC

CECC aligns stable representations from the same class across different source environments while separating different classes.

### CEI

CEI recombines source-observed environmental factors from same-class donors in representation space, encouraging robustness to unseen factor compositions.

See [Method details](docs/METHOD.md) for the mathematical formulation and training objectives.

## Installation

```bash
conda env create -f environment.yml
conda activate difter
```

or:

```bash
pip install -r requirements.txt
```

## Data Preparation

DIFTER expects flow-level traffic samples to be converted into traffic windows containing traffic tokens, packet timing, packet length, and packet-boundary information.

The repository does not redistribute the benchmark datasets. Prepare the datasets according to [the data format](docs/DATA_FORMAT.md).

```bash
python scripts/preprocess.py --config configs/example_dataset.yaml
```

## Training

```bash
python scripts/train.py \
    --config configs/difter.yaml \
    --dataset-config configs/example_dataset.yaml
```

Model selection is based on source-validation Macro-F1; target data are reserved for final evaluation.

## Evaluation

```bash
python scripts/evaluate.py --config configs/difter.yaml --dataset-config configs/example_dataset.yaml --split source_test --checkpoint checkpoints/best.pt
python scripts/evaluate.py --config configs/difter.yaml --dataset-config configs/example_dataset.yaml --split target_test --checkpoint checkpoints/best.pt
```

The same source-selected checkpoint is used for source-test and target evaluation.

## Inference

```bash
python scripts/infer.py --config configs/difter.yaml --dataset-config configs/example_dataset.yaml --input path/to/processed_windows.tsv --checkpoint checkpoints/best.pt
```

Inference retains HPTF, masked window aggregation, the stable projector, and the classifier.

## Project Structure

```text
DIFTER/
├── assets/          # Framework figure
├── configs/         # Model and dataset configurations
├── data/            # Dataset schema and placeholders
├── difter/
│   ├── data/        # Data loading and traffic-window construction
│   ├── models/      # HPTF, CCIF, CECC, CEI, classifier
│   ├── losses/      # Training objectives
│   ├── training/    # Training and scheduling
│   └── evaluation/  # Metrics and evaluation
├── docs/            # Method, data format, reproducibility
├── scripts/         # Training/evaluation/inference entry points
└── tests/           # Unit tests
```

For deterministic setup and configuration details, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Citation

If you find this repository useful, please cite the paper. Citation metadata will be updated upon publication; the current software record is available in [`CITATION.cff`](CITATION.cff).
