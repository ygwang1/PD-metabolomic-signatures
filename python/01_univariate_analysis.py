#!/usr/bin/env python3
"""
Univariate analysis for PD metabolomics.
Includes: fold change with bootstrap CI, logistic regression with covariate
adjustment, and volcano plot visualization.

Reference: "Metabolomic signatures for diagnosis and clinical severity
            in Parkinson's disease"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, mannwhitneyu

from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 66
N_BOOTSTRAP = 1000
CI_LEVEL = 95

# =============================================================================
# 1. Data loading
# =============================================================================
main_data = pd.read_csv(
    "data/example_data.csv", sep=",", encoding="UTF-8", header=0
)
all_met = pd.read_csv("data/metabolite_list.csv", sep=",", header=0)
all_met = all_met["metabolite"].to_list()

# Split by cohort
main_data_c1 = main_data[main_data["cohort"] == "cohort1"]
main_data_c2 = main_data[main_data["cohort"] == "cohort2"]
print(f"Cohort 1: {main_data_c1.shape}, Cohort 2: {main_data_c2.shape}")

# =============================================================================
# 2. Log10 transform and Z-score standardization (per cohort)
# =============================================================================
main_data_log10_c1 = main_data_c1.copy()
main_data_log10_c2 = main_data_c2.copy()

for col in all_met:
    for df in [main_data_log10_c1, main_data_log10_c2]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df[col] = np.log10(df[col])

# Z-score standardization
main_data_log10_z_c1 = main_data_log10_c1.copy()
main_data_log10_z_c2 = main_data_log10_c2.copy()
scaler = StandardScaler()

main_data_log10_z_c1[all_met] = scaler.fit_transform(
    main_data_log10_z_c1[all_met]
)
main_data_log10_z_c2[all_met] = scaler.fit_transform(
    main_data_log10_z_c2[all_met]
)

# Encode binary variables
for df in [main_data_c1, main_data_c2, main_data_log10_z_c1, main_data_log10_z_c2]:
    df["diagnosis"] = df["diagnosis"].map({"NC": 0, "PD": 1})
    df["sex"] = df["sex"].map({"female": 0, "male": 1})

# =============================================================================
# 3. Fold change analysis with bootstrap CI
# =============================================================================
def calculate_fold_change(data, group_col, metabolite_list,
                          n_bootstrap=1000, ci=95):
    """
    Calculate fold change (PD vs NC) with bootstrap confidence intervals
    and statistical tests (t-test, Mann-Whitney U).

    Parameters
    ----------
    data : pd.DataFrame
        Contains grouping column and metabolite columns.
    group_col : str
        Column name for the binary group variable.
    metabolite_list : list
        List of metabolite column names.
    n_bootstrap : int
        Number of bootstrap resampling iterations.
    ci : float
        Confidence interval level (default 95 for 2.5%-97.5%).

    Returns
    -------
    pd.DataFrame with fold change, log2FC, p-values, FDR-adjusted p-values,
    and bootstrap confidence intervals.
    """
    results = []
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Group variable must have exactly two unique values")

    alpha_lower = (100 - ci) / 2 / 100
    alpha_upper = 1 - alpha_lower

    for metabolite in metabolite_list:
        group1_data = data.loc[data[group_col] == 1, metabolite].dropna()  #missing values ​​have already been handled, so there are no NAs.
        group2_data = data.loc[data[group_col] == 0, metabolite].dropna()  #adjust based on the actual code

        if len(group1_data) == 0 or len(group2_data) == 0:
            results.append({
                "variable": metabolite,
                "fold_change": np.nan,
                "log2FoldChange": np.nan,
                "p_value_ttest": np.nan,
                "p_value_mannwhitneyu": np.nan,
                "fold_change_ci_lower": np.nan,
                "fold_change_ci_upper": np.nan,
            })
            continue

        group1_mean = group1_data.mean()
        group2_mean = group2_data.mean()

        fold_change = group1_mean / group2_mean
        log2_fc = np.log2(fold_change)

        # Statistical tests
        _, p_ttest = ttest_ind(group1_data, group2_data, nan_policy="omit")
        _, p_mwu = mannwhitneyu(
            group1_data, group2_data, alternative="two-sided"
        )

        # Bootstrap CI for fold change
        bootstrap_fcs = []
        n1, n2 = len(group1_data), len(group2_data)
        np.random.seed(RANDOM_STATE)
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1_data, size=n1, replace=True)
            sample2 = np.random.choice(group2_data, size=n2, replace=True)
            bootstrap_fcs.append(sample1.mean() / sample2.mean())

        ci_lower = np.percentile(bootstrap_fcs, alpha_lower * 100)
        ci_upper = np.percentile(bootstrap_fcs, alpha_upper * 100)

        results.append({
            "variable": metabolite,
            "fold_change": round(fold_change, 2),
            "log2FoldChange": round(log2_fc, 2),
            "p_value_ttest": p_ttest,
            "p_value_mannwhitneyu": p_mwu,
            "fold_change_ci_lower": round(ci_lower, 2),
            "fold_change_ci_upper": round(ci_upper, 2),
            "ci_crosses_1": 1 if (ci_lower <= 1 <= ci_upper) else 0,
        })

    results_df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    results_df["adj_p_value_ttest"] = multipletests(
        results_df["p_value_ttest"], method="fdr_bh"
    )[1]
    results_df["adj_p_value_mannwhitneyu"] = multipletests(
        results_df["p_value_mannwhitneyu"], method="fdr_bh"
    )[1]

    return results_df


# Run fold change analysis per cohort
result_c1 = calculate_fold_change(
    data=main_data_c1, group_col="diagnosis",
    metabolite_list=all_met, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL
)
result_c1.to_csv("fold_change_results_cohort1.csv", index=False)
print(f"Cohort 1 fold change: {len(result_c1)} metabolites processed")

result_c2 = calculate_fold_change(
    data=main_data_c2, group_col="diagnosis",
    metabolite_list=all_met, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL
)
result_c2.to_csv("fold_change_results_cohort2.csv", index=False)
print(f"Cohort 2 fold change: {len(result_c2)} metabolites processed")

# =============================================================================
# 4. Logistic regression with covariate adjustment
# =============================================================================
def logistic_regression(data, confounders, group_col, metabolite_list):
    """
    Multiple logistic regression per metabolite, adjusting for covariates.

    Returns ORs, 95% CIs, p-values, and FDR-adjusted p-values.
    """
    all_results = []

    for metab in metabolite_list:
        X = data[confounders + [metab]].copy()
        valid_idx = X.dropna().index
        X_clean = X.loc[valid_idx]
        y_clean = data.loc[valid_idx, group_col]

        X_clean = sm.add_constant(X_clean)

        try:
            model = sm.Logit(y_clean, X_clean).fit(disp=0)
        except Exception as e:
            print(f"  Model fitting failed for {metab}: {e}")
            continue

        params = model.params
        conf_int = model.conf_int()
        p_values = model.pvalues

        # Exclude intercept
        non_const_mask = params.index != "const"
        params = params[non_const_mask]
        conf_int = conf_int.loc[non_const_mask]
        p_values = p_values[non_const_mask]

        # Odds ratios and CIs
        odds_ratios = np.exp(params)
        ci_lower = np.exp(conf_int.iloc[:, 0])
        ci_upper = np.exp(conf_int.iloc[:, 1])

        result_df = pd.DataFrame({
            "metabolite": metab,
            "n": len(X_clean),
            "variable": params.index,
            "beta": params.values,
            "OR": odds_ratios.values,
            "CI_lower": ci_lower.values,
            "CI_upper": ci_upper.values,
            "p_value": p_values.values,
        })
        all_results.append(result_df)

    if not all_results:
        raise ValueError("No metabolite models were successfully fitted")

    combined = pd.concat(all_results, ignore_index=True)
    _, q_values, _, _ = multipletests(
        combined["p_value"].dropna(), method="fdr_bh"
    )
    combined["adj_p_value"] = np.nan
    combined.loc[combined["p_value"].notna(), "adj_p_value"] = q_values

    return combined


# Covariate sets for sensitivity analysis
covariate_sets = {
    "basic": ["age", "sex", "bmi", "HBP", "diabetes"],
    "modelA": ["age", "sex", "bmi", "HBP", "diabetes", "smoke", "alcohol"],
    "modelB": [
        "age", "sex", "bmi", "HBP", "diabetes", "smoke", "alcohol",
        "CRP", "ALB", "CST3"
    ],
    "modelC": [
        "age", "sex", "bmi", "HBP", "diabetes", "smoke", "alcohol",
        "CRP", "ALB", "CST3", "B2M", "Creatinine_cov", "Bilirubin_cov"
    ],
}

datasets = {
    "cohort1": main_data_log10_z_c1,
    "cohort2": main_data_log10_z_c2,
}

diagnosis_col = "diagnosis"

for dataset_name, data_tmp in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    for cov_set_name, covariates in covariate_sets.items():
        print(f"  Covariate set: {cov_set_name}")
        logistic_results = logistic_regression(
            data_tmp, covariates, diagnosis_col, all_met
        )
        # Filter to metabolite-only rows
        logistic_res = logistic_results[
            ~logistic_results["variable"].isin(covariates)
        ]
        logistic_results.to_csv(
            f"logistic_regression_{cov_set_name}_{dataset_name}_full.csv",
            index=False
        )
        logistic_res.to_csv(
            f"logistic_regression_{cov_set_name}_{dataset_name}_metabolites.csv",
            index=False
        )
        print(f"    Done: {len(logistic_res)} metabolite records saved")

# =============================================================================
# 5. Volcano plot
# =============================================================================
sns.reset_defaults()
sns.set(font_scale=1.1)
sns.set_style("white")

# Load combined fold-change + FDR + VIP data for plotting
# This file should be prepared by merging fold change results with VIP scores
volcano_data = pd.read_table(
    "fold_change_results_trim_c1.txt", sep="\t", encoding="UTF-8", header=0
)

volcano_data["neg_log10_fdr"] = -np.log10(volcano_data["FDR"])

# Color gradient by log2 fold change
norm = plt.Normalize(
    volcano_data["log2FoldChange"].min(),
    volcano_data["log2FoldChange"].max()
)
colors = plt.cm.bwr(norm(volcano_data["log2FoldChange"]))

fig, ax = plt.subplots(figsize=(7, 10))
sc = ax.scatter(
    volcano_data["log2FoldChange"],
    volcano_data["neg_log10_fdr"],
    s=volcano_data["VIP"] * 40,
    c=colors,
    alpha=0.8,
    edgecolor="black"
)

# Label significant metabolites (FDR < 0.05 and |log2FC| > 1)
texts = []
for i, row in volcano_data.iterrows():
    if row["FDR"] < 0.05 and abs(row["log2FoldChange"]) > 1:
        texts.append(
            ax.text(
                row["log2FoldChange"], row["neg_log10_fdr"],
                row["variable"],
                fontsize=8, ha="left", va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="gray",
                    facecolor="white", alpha=0.6
                )
            )
        )

# Adjust label positions to avoid overlap
adjust_text(
    texts,
    expand_text=(4, 4),
    force_text=(0.5, 0.8),
    force_points=(0.3, 0.5),
    only_move={"points": "xy", "texts": "xy"},
    ax=ax
)

# Reference lines
ax.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=1)
ax.axvline(1, color="green", linestyle="--", linewidth=1)
ax.axvline(-1, color="green", linestyle="--", linewidth=1)

ax.set_xlabel("log$_{2}$(Fold Change)", fontsize=14)
ax.set_ylabel("-log$_{10}$(Adjusted p-value)", fontsize=14)

# VIP legend
vip_legend = [
    plt.scatter(
        [], [], s=vip * 40, c="grey", alpha=0.5, label=f"{vip}"
    )
    for vip in [1.0, 2.0, 3.0, 4.0]
]
ax.legend(
    handles=vip_legend, loc="center left",
    bbox_to_anchor=(1.02, 0.25), title="VIP"
)

# Colorbar for log2 fold change
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap="bwr"),
    ax=ax, pad=0.06, fraction=0.02, shrink=1
)
cbar.set_label("log$_{2}$(Fold Change)", fontsize=14)

plt.tight_layout()
plt.savefig("fig_volcano_plot.pdf", dpi=600)
plt.show()
print("\nUnivariate analysis complete.")
