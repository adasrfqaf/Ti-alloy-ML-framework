# Ti-Alloy-ML-Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22299875.svg)](https://doi.org/10.5281/zenodo.22299875)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A machine learning framework for predicting hydrogen storage performance of Ti-based alloys from small experimental datasets.

##  Overview

This repository contains the complete code and data for the paper:
> **Predicting Ti-Based Hydrogen Storage Alloy Performance from Small Experimental Datasets: A Machine Learning Framework with Phase‑stratified Modeling and Adaptive Augmentation**

The framework integrates phase-stratified modeling, adaptive data augmentation, and hierarchical multi-model screening to address the dual challenges of multiphase heterogeneity and small-sample overfitting.

##  Key Features

- **Phase-stratified modeling** for BCC, TiFe, and C14 Laves phases
- **Four data augmentation strategies**: GN, SMOTER, SMOTER-GN, and SMOGN
- **Hierarchical model screening** with XGBoost, GBDT, RF, SVR, LGBM, and MLP
- **SHAP-based feature attribution** for model interpretability
- **Candidate alloy screening** for experimental guidance

##  Repository Structure

```
Ti‑alloy‑ML‑framework/
├── Unaugmented Dataset/          # Raw experimental data
├── Adaptive Dataset Augmentation/ # Augmented datasets
├── Global mixed training/        # Baseline global models
├── Material Screening/           # Candidate alloy screening scripts
├── experimental data figure/     # Visualization scripts
├── results/                      # Auto‑generated outputs, model predictions (git‑ignored)
├── figures/                      # Auto‑generated figure outputs (git‑ignored)
├── requirements.txt              # Fixed‑version python dependencies
├── LICENSE                       # MIT License
└── README.md
```


## 🚀 Getting Started
### Prerequisites
- Python **3.11**
- Required packages:
  - `numpy`, `pandas` – data manipulation
  - `scipy` – numerical computation
  - `scikit‑learn` – provides Random Forest (RF), Support Vector Regression (SVR), Multi‑layer Perceptron (MLP), Gradient Boosting Decision Tree (GBDT), and PCA
  - `xgboost` – XGBoost model
  - `shap` – SHAP feature attribution
  - `imbalanced‑learn`, `smogn` – implementations for data augmentation strategies

> Note: Trained model files (`*.pkl`) will be generated locally after running scripts and are not tracked by Git.

### Installation
```bash
git clone https://github.com/adasrfqaf/Ti‑alloy‑ML‑framework.git
cd Ti‑alloy‑ML‑framework
pip install -r requirements.txt
```

### Quick Example
```python
# Example: Train a model on BCC phase data
from models import train_phase_model
model = train_phase_model(phase="BCC", augmentation="SMOGN")
```

##  Results

Under the current dataset split, the test-set $R^2$ exceeds 0.93 for all three phases, with overfitting gap $\Delta R^2$ maintained below 0.05.

##  Citation

If you use this code in your research, please cite:

```bibtex
@software{Zhang_Ti-Alloy-ML-Framework_2026,
  author = {Zhang, Yiming},
  title = {Ti-Alloy-ML-Framework},
  version = {1.0.0},
  year = {2026},
  doi = {10.5281/zenodo.22299875},
  url = {https://github.com/adasrfqaf/Ti-alloy-ML-framework}
}
```

##  License

This project is licensed under the MIT License ‑ see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

This work was supported by [your funding information].
