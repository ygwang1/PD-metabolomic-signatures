#!/usr/bin/env python3
"""
Machine learning model training and evaluation for PD metabolomics.

Workflow:
  1. Load LASSO-selected metabolites and log10-transformed data
  2. Train LR, SVM, XGBoost with nested cross-validation
  3. Plot ROC curves (combined and per-cohort)
  4. DeLong test for model comparison
  5. External evaluation on essential tremor (ET) data
  6. Single-metabolite diagnostic performance

Reference: "Metabolomic signatures for diagnosis and clinical severity
            in Parkinson's disease"
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

import joblib
from collections import Counter

# Local utilities
from utils import (
    confusion_elements, compute_metrics, eval_binary,
    fast_delong, delong_roc_test,
)

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 66
OUTER_CV = 5
INNER_CV = 5

# Load stable metabolites selected by LASSO stability analysis
selected_metabolites = pd.read_csv(
    "data/selected_metabolites.csv"
)["metabolite"].tolist()
N_METABOLITES = len(selected_metabolites)
print(f"Selected metabolites (n={N_METABOLITES}): {selected_metabolites}")

# =============================================================================
# 1. Load data
# =============================================================================
train = pd.read_csv("train_data_log10.csv", sep=",", encoding="UTF-8", header=0)
test = pd.read_csv("test_data_log10.csv", sep=",", encoding="UTF-8", header=0)

X_train = train[selected_metabolites]
y_train = train["diagnosis"]
X_test = test[selected_metabolites]
y_test = test["diagnosis"]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# =============================================================================
# 2. Define models and hyperparameter grids
# =============================================================================
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
])

pipe_xgb = Pipeline([
    ("clf", XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss"
    )),
])

param_grid_lr = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__solver": ["liblinear"],
}

param_grid_svm = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.01, 0.1],
    "clf__kernel": ["rbf"],
}

param_grid_xgb = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [2, 3],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__reg_lambda": [1, 2],
}

# =============================================================================
# 3. Nested cross-validation
# =============================================================================
def nested_cv(model_pipe, param_grid, X, y, model_name,
              outer=5, inner=5):
    """
    Nested cross-validation with hyperparameter tuning.

    Returns dict with ROC data, CV metrics, test metrics, best params,
    and test set predicted probabilities.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer, shuffle=True, random_state=RANDOM_STATE
    )

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold_rocs = []
    best_params_list = []

    metrics = {
        "AUC": [], "Accuracy": [], "Precision": [],
        "Recall": [], "F1": [],
    }

    for i, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y)
    ):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        inner_cv = StratifiedKFold(
            n_splits=inner, shuffle=True, random_state=RANDOM_STATE
        )
        grid = GridSearchCV(
            model_pipe, param_grid,
            cv=inner_cv, scoring="roc_auc", n_jobs=-1,
        )
        grid.fit(X_tr, y_tr)

        best_model = grid.best_estimator_
        best_params_list.append(grid.best_params_)

        y_prob = best_model.predict_proba(X_val)[:, 1]
        y_pred = best_model.predict(X_val)

        metrics["AUC"].append(roc_auc_score(y_val, y_prob))
        metrics["Accuracy"].append(accuracy_score(y_val, y_pred))

        # ROC per fold
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        fold_rocs.append((fpr, tpr, roc_auc_score(y_val, y_prob)))

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y_val, y_prob))

    # Summarize CV metrics
    cv_summary = {
        k: (np.mean(v), np.std(v)) for k, v in metrics.items()
    }

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    # Final model fitted on all training data
    final_model = GridSearchCV(
        model_pipe, param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1,
    )
    final_model.fit(X, y)
    best_estimator = final_model.best_estimator_

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(
        best_estimator,
        f"saved_models/{model_name}_final_model.pkl",
    )

    # Test set evaluation
    y_prob_test = best_estimator.predict_proba(X_test)[:, 1]
    y_pred_test = best_estimator.predict(X_test)

    test_metrics = {
        "AUC": roc_auc_score(y_test, y_prob_test),
        "Accuracy": accuracy_score(y_test, y_pred_test),
    }
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

    return {
        "roc": {
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "fpr_test": fpr_test,
            "tpr_test": tpr_test,
            "fold_rocs": fold_rocs,
        },
        "cv_metrics": cv_summary,
        "test_metrics": test_metrics,
        "best_params": best_params_list,
        "y_prob_test": y_prob_test,
        "best_estimator": best_estimator,
    }


# Run models
models_config = {
    "LogisticRegression": (pipe_lr, param_grid_lr),
    "SVM": (pipe_svm, param_grid_svm),
    "XGBoost": (pipe_xgb, param_grid_xgb),
}

results = {}
for name, (pipe, grid) in models_config.items():
    print(f"\nRunning {name}...")
    results[name] = nested_cv(pipe, grid, X_train, y_train, name,
                              outer=OUTER_CV, inner=INNER_CV)

# =============================================================================
# 4. Print metrics summary
# =============================================================================
for name, res in results.items():
    print(f"\n===== {name} =====")
    print("CV Metrics:")
    for k, (mean, std) in res["cv_metrics"].items():
        print(f"  {k}: {mean:.4f} +/- {std:.4f}")
    print("Test Metrics:")
    for k, v in res["test_metrics"].items():
        print(f"  {k}: {v:.4f}")

    counter = Counter([str(p) for p in res["best_params"]])
    print("Most common params:", counter.most_common(1)[0][0])

# Build summary table
summary_rows = []
for name, res in results.items():
    row = {"Model": name}
    for metric, (mean, std) in res["cv_metrics"].items():
        row[f"CV_{metric}"] = f"{mean:.4f} +/- {std:.4f}"
    for metric, value in res["test_metrics"].items():
        row[f"Test_{metric}"] = f"{value:.4f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("\n", summary_df)

# Save prediction probabilities for model comparison
model_probs = {}
for model_name, res in results.items():
    model_probs[model_name] = {
        "y_true": y_test.values,
        "y_prob": res["y_prob_test"],
    }

with open(f"model_probs_{N_METABOLITES}met.pkl", "wb") as f:
    pickle.dump(model_probs, f)

# =============================================================================
# 5. Figure: ROC curves (training CV + test)
# =============================================================================
sns.reset_defaults()
sns.set(font_scale=1.2)
sns.set_style("white")

for name, res in results.items():
    fig, ax = plt.subplots(figsize=(5, 5))

    # Per-fold ROC (training CV)
    for i, (fpr, tpr, auc_val) in enumerate(res["roc"]["fold_rocs"]):
        ax.plot(
            fpr, tpr, alpha=0.3,
            label=f"Fold {i+1} AUC={auc_val:.2f}"
        )

    # Mean ROC
    mean_fpr = res["roc"]["mean_fpr"]
    mean_tpr = res["roc"]["mean_tpr"]
    std_tpr = res["roc"]["std_tpr"]
    mean_auc, std_auc = res["cv_metrics"]["AUC"]

    ax.plot(
        mean_fpr, mean_tpr, linewidth=2, color="b",
        label=f"Mean CV AUC={mean_auc:.2f}+/-{std_auc:.2f}"
    )
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.2, color="blue",
    )

    # Test ROC
    ax.plot(
        res["roc"]["fpr_test"], res["roc"]["tpr_test"],
        linestyle="--", linewidth=2, color="r",
        label=f"Test AUC={res['test_metrics']['AUC']:.2f}"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        f"fig_roc_{name.lower()}.pdf", dpi=300
    )
    plt.show()

# =============================================================================
# 6. DeLong test for model comparison
# =============================================================================
print("\n===== DeLong Test =====")
model_names = list(results.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        p = delong_roc_test(
            y_test,
            results[m1]["y_prob_test"],
            results[m2]["y_prob_test"],
        )
        print(f"{m1} vs {m2}: p = {p:.4f}")

# =============================================================================
# 7. Per-cohort evaluation
# =============================================================================
print("\n===== Per-Cohort Evaluation =====")

test_df = pd.read_csv(
    "test_data_log10.csv", sep=",", encoding="UTF-8", header=0
)

test_c1 = test_df[test_df["cohort"] == "cohort1"].copy()
test_c2 = test_df[test_df["cohort"] == "cohort2"].copy()

X_test_c1 = test_c1[selected_metabolites]
y_test_c1 = test_c1["diagnosis"]
X_test_c2 = test_c2[selected_metabolites]
y_test_c2 = test_c2["diagnosis"]

# Load saved models
loaded_models = {}
for name in models_config:
    model_path = f"saved_models/{name}_final_model.pkl"
    loaded_models[name] = joblib.load(model_path)

# Plot per-cohort ROC
for name, model in loaded_models.items():
    fig, ax = plt.subplots(figsize=(5, 5))

    # Cohort 1
    _, fpr1, tpr1, _ = eval_binary(model, X_test_c1, y_test_c1)
    auc1 = roc_auc_score(y_test_c1, model.predict_proba(X_test_c1)[:, 1])
    ax.plot(
        fpr1, tpr1, linewidth=2.5, color="#E64B35",
        label=f"Cohort 1 (AUC = {auc1:.3f})"
    )

    # Cohort 2
    _, fpr2, tpr2, _ = eval_binary(model, X_test_c2, y_test_c2)
    auc2 = roc_auc_score(y_test_c2, model.predict_proba(X_test_c2)[:, 1])
    ax.plot(
        fpr2, tpr2, linewidth=2.5, linestyle="--", color="#4DBBD5",
        label=f"Cohort 2 (AUC = {auc2:.3f})"
    )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title(name, fontsize=16)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"fig_roc_{name.lower()}_by_cohort.pdf", dpi=300)
    plt.show()

# Per-cohort confusion matrices
for name, model in loaded_models.items():
    for cohort_label, X_ct, y_ct in [
        ("cohort1", X_test_c1, y_test_c1),
        ("cohort2", X_test_c2, y_test_c2),
    ]:
        y_score = model.predict_proba(X_ct)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        cm = pd.crosstab(
            pd.Series(y_ct, name="True"),
            pd.Series(y_pred, name="Predicted"),
        )
        print(f"\n{name} - {cohort_label} confusion matrix:")
        print(cm)

# =============================================================================
# 8. External evaluation: Essential Tremor (ET) data
# =============================================================================
print("\n===== External Evaluation (ET data) =====")

et_data = pd.read_csv(
    "data/example_et_data.csv", sep=",", encoding="UTF-8", header=0
)
# Log10 transform metabolites in ET data
for col in selected_metabolites:
    et_data[col] = pd.to_numeric(
        et_data[col], errors="coerce"
    ).astype("float64")
    et_data[col] = np.log10(et_data[col])

X_et = et_data[selected_metabolites]
y_et = et_data["diagnosis"]  # 0=HC, 1=PD, 2=ET

print(f"ET data: {et_data.shape[0]} samples")
print(f"Diagnosis distribution: {dict(y_et.value_counts())}")


def eval_pairwise_roc(model, X, y, pair, positive_class):
    """Evaluate model for a specific pair of classes."""
    mask = y.isin(pair)
    y_binary = (y[mask] == positive_class).astype(int)
    y_score = model.predict_proba(X[mask])[:, 1]
    auc = roc_auc_score(y_binary, y_score)
    fpr, tpr, _ = roc_curve(y_binary, y_score)
    return auc, fpr, tpr


# Plot pairwise ROC for HC vs PD and PD vs ET
sns.set_style("white")
sns.set_context("talk")

for name, model in loaded_models.items():
    fig, ax = plt.subplots(figsize=(5, 5))

    # HC vs PD
    auc_hc_pd, fpr_01, tpr_01 = eval_pairwise_roc(
        model, X_et, y_et, [0, 1], positive_class=1
    )
    ax.plot(
        fpr_01, tpr_01, linewidth=2.5,
        label=f"HC vs PD (AUC = {auc_hc_pd:.3f})"
    )

    # PD vs ET
    auc_pd_et, fpr_12, tpr_12 = eval_pairwise_roc(
        model, X_et, y_et, [1, 2], positive_class=1
    )
    ax.plot(
        fpr_12, tpr_12, linewidth=2.5, linestyle="--",
        label=f"PD vs ET (AUC = {auc_pd_et:.3f})"
    )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"fig_roc_{name.lower()}_et_pairwise.pdf", dpi=300)
    plt.show()

# Summary table for ET evaluation
et_rows = []
for name, model in loaded_models.items():
    auc_hc_pd, _, _ = eval_pairwise_roc(model, X_et, y_et, [0, 1], 1)
    auc_pd_et, _, _ = eval_pairwise_roc(model, X_et, y_et, [1, 2], 1)
    et_rows.append({
        "Model": name,
        "AUC_HC_vs_PD": auc_hc_pd,
        "AUC_PD_vs_ET": auc_pd_et,
    })

et_df = pd.DataFrame(et_rows)
print("\nET Evaluation Summary:")
print(et_df)

# =============================================================================
# 9. Single-metabolite diagnostic performance
# =============================================================================
print("\n===== Single-Metabolite Analysis =====")

results_single = []

for met in selected_metabolites:
    print(f"\nProcessing: {met}")

    X_tr_met = X_train[[met]].values
    X_te_met = X_test[[met]].values
    y_tr_met = y_train.values
    y_te_met = y_test.values

    models_single = {
        "LR": (Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            )),
        ]), {
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear"],
        }),
        "SVM": (Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                probability=True, random_state=RANDOM_STATE
            )),
        ]), {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__kernel": ["rbf"],
        }),
        "XGBoost": (Pipeline([
            ("clf", XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss"
            )),
        ]), {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [2, 3],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__reg_lambda": [1, 2],
        }),
    }

    for model_abbr, (pipe, param_grid) in models_single.items():
        outer_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        cv_metrics = {
            "AUC": [], "Accuracy": [], "Precision": [],
            "Recall": [], "F1": [],
        }

        for train_idx, val_idx in outer_cv.split(X_tr_met, y_tr_met):
            X_tr, X_val = X_tr_met[train_idx], X_tr_met[val_idx]
            y_tr, y_val = y_tr_met[train_idx], y_tr_met[val_idx]

            inner_cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=RANDOM_STATE
            )
            grid = GridSearchCV(
                pipe, param_grid,
                cv=inner_cv, scoring="roc_auc", n_jobs=-1,
            )
            grid.fit(X_tr, y_tr)

            best_model = grid.best_estimator_
            y_prob = best_model.predict_proba(X_val)[:, 1]
            y_pred = best_model.predict(X_val)

            cv_metrics["AUC"].append(roc_auc_score(y_val, y_prob))
            cv_metrics["Accuracy"].append(accuracy_score(y_val, y_pred))

            fpr, tpr, _ = roc_curve(y_val, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_val, y_prob))

        # Final model and test evaluation
        final_model = GridSearchCV(
            pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )
        final_model.fit(X_tr_met, y_tr_met)
        best_estimator = final_model.best_estimator_

        y_prob_test = best_estimator.predict_proba(X_te_met)[:, 1]
        y_pred_test = best_estimator.predict(X_te_met)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)

        test_metrics = {
            "AUC": roc_auc_score(y_te_met, y_prob_test),
            "Accuracy": accuracy_score(y_te_met, y_pred_test),
        }

        results_single.append({
            "Metabolite": met,
            "Model": model_abbr,
            "CV_AUC_mean": np.mean(cv_metrics["AUC"]),
            "CV_Accuracy_mean": np.mean(cv_metrics["Accuracy"]),
            "Test_AUC": test_metrics["AUC"],
            "Test_Accuracy": test_metrics["Accuracy"],
        })

        # ROC plot per metabolite-model
        safe_name = met.replace("/", "_").replace(" ", "_")[:30]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            mean_fpr, mean_tpr, color="b", linewidth=2,
            label=f"CV AUC={np.mean(cv_metrics['AUC']):.2f}"
        )
        ax.fill_between(
            mean_fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.2, color="blue",
        )
        fpr_test, tpr_test, _ = roc_curve(y_te_met, y_prob_test)
        ax.plot(
            fpr_test, tpr_test, linestyle="--", color="r",
            linewidth=2, label=f"Test AUC={test_metrics['AUC']:.2f}"
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{met} ({model_abbr})", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        fig.savefig(
            f"fig_roc_single_{safe_name}_{model_abbr.lower()}.pdf",
            dpi=300,
        )
        plt.close(fig)

# Save single-metabolite summary
single_summary = pd.DataFrame(results_single)
single_summary.to_csv("single_metabolite_performance.csv", index=False)
print("\nSingle-metabolite performance summary saved.")
print(single_summary)

# =============================================================================
# 10. Radar chart: model performance by drug-naïve vs medicated subgroups
# =============================================================================
from math import pi

perf = pd.read_csv("performance.txt", sep="\t")

groups = ["Test set-drug-naïve", "Test set-medicated"]
metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1-score"]

colors = {
    "Logistic Regression": "#8C6BB1",
    "Support Vector Machine": "#FB8072",
    "eXtreme Gradient Boosting": "#80B1D3",
}
linestyles = {
    "Logistic Regression": "-",
    "Support Vector Machine": "--",
    "eXtreme Gradient Boosting": "-.",
}
markers = {
    "Logistic Regression": "o",
    "Support Vector Machine": "s",
    "eXtreme Gradient Boosting": "^",
}

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

fig, axes = plt.subplots(
    1, 2, figsize=(14, 7),
    subplot_kw=dict(polar=True),
    gridspec_kw={"wspace": 0.35},
)

for ax, group in zip(axes, groups):
    group_data = perf[perf["Metabolite"] == group]

    for _, row in group_data.iterrows():
        model = row["Model"]
        values = row[metrics].values.tolist()
        values += values[:1]
        ax.plot(
            angles, values,
            color=colors[model],
            linestyle=linestyles[model],
            marker=markers[model],
            markersize=5,
            linewidth=2,
            alpha=1,
            label=model,
        )
        ax.fill(angles, values, alpha=0.06, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0"], fontsize=10)
    ax.set_title(group, fontsize=13, pad=15, weight="semibold")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["polar"].set_color("#cccccc")

# Unified legend
handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))
fig.legend(
    unique.values(), unique.keys(),
    title="Model",
    loc="center right",
    bbox_to_anchor=(0.6, 0.8),
    fontsize=10,
    title_fontsize=11,
    frameon=False,
)

fig.savefig("fig_model_performance_radar.pdf", dpi=300)
plt.show()
print("\nRadar chart saved.")

print("\nModel training and evaluation complete.")
