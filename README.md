# Metabolomic Signatures for Diagnosis and Clinical Severity in Parkinson's Disease

Analysis code for the paper "Metabolomic signatures for diagnosis and clinical severity in Parkinson's disease."

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── R/
│   ├── lasso_stability_selection.R    # LASSO feature selection with stability analysis
│   └── pca_oplsda_analysis.R         # PCA and OPLS-DA multivariate analysis
├── python/
│   ├── 00_data_preparation.py         # Data loading, train/test split, log10 transform
│   ├── 01_univariate_analysis.py      # Fold change, logistic regression, volcano plot
│   ├── 02_model_training_evaluation.py # Nested CV, ROC, DeLong test, single-metabolite
│   └── utils.py                       # Shared evaluation utilities
└── data/
    ├── example_data.csv               # Example dataset (see Data Format below)
    ├── example_et_data.csv             # Example external validation data (HC/PD/ET)
    ├── metabolite_list.csv            # List of 749 QC-passed metabolites
    └── selected_metabolites.csv       # LASSO-stability-selected metabolites
```

## Requirements

### R (>= 4.4.1)
```r
install.packages(c(
  "glmnet", "ggplot2", "reshape2", "pheatmap", "RColorBrewer",
  "dplyr", "psych", "fastDummies", "data.table",
  "ropls", "factoextra", "vegan", "ggpubr", "ggprism",
  "patchwork", "tidyr"
))
```

### Python (>= 3.12)
```
pip install -r requirements.txt
```

## Data Format

### Input data (`data/example_data.csv`)

The main metabolomics dataset should be a CSV file with the following structure:

- **Rows**: One row per sample
- **Columns**:
  - **Sample identifiers and metadata**: `ID`, `sex`, `age`, `bmi`
  - **Clinical variables**: `diagnosis` (values: `NC` for healthy controls, `PD` for Parkinson's disease), `cohort` (values: `cohort1`, `cohort2`)
  - **Covariates** (optional, used in adjusted analyses): `HBP`, `diabetes`, `smoke`, `alcohol`, `CRP`, `ALB`, `CST3`, `B2M`, `Creatinine_cov`, `Bilirubin_cov`
  - **Metabolite columns**: One column per metabolite (named by metabolite name), containing raw (untransformed) intensity values

Example data files are provided in `data/` for testing the pipeline. Replace them with your own data following the same format.

### External validation data (`data/example_et_data.csv`)

For the external ET (essential tremor) evaluation, provide a CSV with the same metabolite columns plus:
- `diagnosis`: `0` = HC, `1` = PD, `2` = ET
- Clinical variables as in the main dataset

### Metabolite list (`data/metabolite_list.csv`)

A single-column CSV with header `metabolite` listing all QC-passed metabolite names to include in the analysis.

## Analysis Workflow

Run scripts in the following order from the repository root:

1. **`python/00_data_preparation.py`** — Loads raw data, splits into train/test (80/20, stratified by diagnosis), applies log10 transformation, and saves prepared datasets (`train_data.csv`, `test_data.csv`).

2. **`R/lasso_stability_selection.R`** — Performs LASSO regression with 10-fold CV and stability selection (200 subsampling iterations, 50% subsampling, selection probability ≥ 0.85). Clinical covariates (age, sex, BMI, hypertension, diabetes) are included as unpenalized variables. Outputs:
   - `lasso_selected_1se.csv` — Metabolites selected by lambda.1se
   - `lasso_stability_probabilities.csv` — Selection probabilities for all metabolites
   - `data/selected_metabolites.csv` — Stable metabolites (selection probability ≥ 0.85)
   - `fig_stability_selection_probability.pdf`
   - `fig_lasso_cv_curve.pdf`
   - `fig_lasso_coefficient_path.pdf`

3. **`R/pca_oplsda_analysis.R`** — Runs PCA (scree plot, score plot) and OPLS-DA (score plot, permutation test with n = 999, VIP with 1000-iteration bootstrap stability). Input data should be log10-transformed and Z-score standardized within the cohort of interest. Outputs:
   - `oplsda_vip_results.csv` — VIP scores with bootstrap confidence intervals and stability metrics
   - `fig_pca_scree.pdf`, `fig_pca_score.pdf`
   - `fig_oplsda_score.pdf`, `fig_oplsda_permutation_test.pdf`
   - `fig_vip_heatmap.pdf`

4. **`python/01_univariate_analysis.py`** — Calculates fold change with bootstrap CI (1000 iterations) for each cohort, runs logistic regression with four progressively expanded covariate adjustment sets (basic, Model A/B/C), and generates the volcano plot. Outputs:
   - `fold_change_results_cohort1.csv`, `fold_change_results_cohort2.csv`
   - `logistic_regression_*_cohort*.csv`
   - `fig_volcano_plot.pdf`

5. **`python/02_model_training_evaluation.py`** — Trains Logistic Regression, SVM, and XGBoost models using nested 5-fold CV with hyperparameter optimization. Evaluates on the independent test set, performs DeLong's test, conducts per-cohort evaluation and external ET validation, and assesses single-metabolite diagnostic performance. Outputs:
   - `fig_roc_*.pdf` — ROC curves for each model
   - `single_metabolite_performance.csv`
   - Trained models saved in `saved_models/`

## Data Availability
Raw data are not included in this repository due to patient privacy protections. Anonymized data are available from the corresponding author upon reasonable request, as stated in the manuscript.

## Citation

If you use this code, please cite the corresponding paper.
Wang Y, Xiang Y, Huang X, et al. Metabolomic signatures for diagnosis and clinical severity in Parkinson's disease. EBioMedicine. Published online July 14, 2026. doi:10.1016/j.ebiom.2026.106383
