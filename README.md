# MedGCN – NHANES Diabetes Classification (Code Repository)

This repository contains an implementation for **diabetes classification** on the  
**NHANES (National Health and Nutrition Examination Survey)** dataset using graph-based models.

> **Scope of this README**  
> This document describes **dataset usage, project structure, and how to run the code only**.  
> Model theory and mathematical formulations are intentionally excluded.

---

## Dataset

### Data Source
- Dataset: **NHANES (National Health and Nutrition Examination Survey)**
- Publisher: CDC (Centers for Disease Control and Prevention)
- Kaggle mirror:  
  https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey

### Unit of Analysis
- **Patient-level**
- Patient identifier: `SEQN`

### Data Files Used

All raw CSV files are stored in:

```
data/
```

| File | Description |
|-----|------------|
| `demographic.csv` | Demographic information |
| `labs.csv` | Laboratory test results |
| `diet.csv` | Dietary intake data |
| `examination.csv` | Physical examination measurements |
| `medications.csv` | Medication usage |
| `questionnaire.csv` | Health questionnaire data |

### Label Definition
- Source file: `questionnaire.csv`
- Column used: `DIQ010`
- Task: **Binary classification**
  - Diabetes: Yes / No

---

## Data Processing & Graph Construction

All data processing and graph construction logic is implemented in:

```
preparedata.py
```

### Node Definition
- Each node represents **one patient**

### Node Features
- Feature matrix **X** is constructed from **numeric demographic attributes**
- Z-score normalization is applied
- Missing values are filled with zero

### Multi-Relation Patient Graph

A **multi-relation patient–patient graph** is constructed.  
Each relation corresponds to a medical modality:

| Relation | Source |
|--------|--------|
| labs | `labs.csv` |
| diet | `diet.csv` |
| exam | `examination.csv` |
| meds | `medications.csv` |
| ques | `questionnaire.csv` |

For each relation:
- A k-nearest neighbor graph is built
- `k = 10`
- Similarity metric: **cosine similarity**
- Adjacency matrices are symmetrically normalized

The data loader returns:

```python
X, y, train_idx, val_idx, test_idx, adjs, agg_adj, rel_names
```

---

## Project Structure

```
MedGCN/
├── train_all.py              # Main training and evaluation script
├── preparedata.py            # Data loading and graph construction
├── medgcn_model.py           # MedGCN baseline model
├── medgcn_relatt_model.py    # MedGCN with relation-level attention
├── rgcn_model.py             # R-GCN baseline
├── gat_model.py              # GAT baseline
├── gcn_model.py              # GCN baseline
├── data/                     # NHANES CSV files
├── results/                  # Outputs: plots, logs, result tables
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Models
```bash
python train_all.py
```

The script will:
- Load NHANES data
- Construct patient graphs
- Train models
- Evaluate performance on the test set
- Save results and plots to the `results/` directory

---

## Outputs

All outputs are saved under:

```
results/
```

Including:
- Training / validation curves
- ROC curves
- Test performance tables (`.csv`)

---

## References

- MedGCN original implementation (code reference):  
  https://github.com/mocherson/MedGCN

- NHANES dataset (Kaggle mirror):  
  https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey
