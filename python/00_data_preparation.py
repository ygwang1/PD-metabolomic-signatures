#!/usr/bin/env python3
"""
Data preparation for PD metabolomics model training.

Steps:
  1. Load cleaned metabolomics data
  2. Encode categorical variables
  3. Stratified train/test split (80/20)
  4. Log10-transform metabolite intensities
  5. Save training and test sets for downstream analysis

Reference: "Metabolomic signatures for diagnosis and clinical severity
            in Parkinson's disease"
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 66
TEST_SIZE = 0.2

# =============================================================================
# 1. Load data
# =============================================================================
main_data = pd.read_csv(
    "data/example_data.csv", sep=",", encoding="UTF-8", header=0
)
all_met = pd.read_csv("data/metabolite_list.csv", sep=",", header=0)
all_met = all_met["metabolite"].to_list()

print(f"Loaded data: {main_data.shape[0]} samples, {len(all_met)} metabolites")
print(f"Cohort distribution:\n{main_data['cohort'].value_counts()}")

# =============================================================================
# 2. Encode categorical variables
# =============================================================================
main_data["diagnosis"] = main_data["diagnosis"].map({"NC": 0, "PD": 1})
main_data["sex"] = main_data["sex"].map({"female": 0, "male": 1})

# =============================================================================
# 3. Stratified train/test split
# =============================================================================
y = main_data["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    main_data,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Verify split
train_cohort_counts = X_train["cohort"].value_counts()
test_cohort_counts = X_test["cohort"].value_counts()
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"  Diagnosis: {dict(X_train['diagnosis'].value_counts())}")
print(f"  Cohort: {dict(train_cohort_counts)}")
print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  Diagnosis: {dict(X_test['diagnosis'].value_counts())}")
print(f"  Cohort: {dict(test_cohort_counts)}")

# Verify per-cohort balance in training set
for cohort in ["cohort1", "cohort2"]:
    subset = X_train[X_train["cohort"] == cohort]
    print(f"  {cohort} diagnosis: {dict(subset['diagnosis'].value_counts())}")

# =============================================================================
# 4. Save train/test sets (before log-transform, for R compatibility)
# =============================================================================
X_train.to_csv("train_data.csv", index=False)
X_test.to_csv("test_data.csv", index=False)
print("\nSaved train_data.csv and test_data.csv")

# =============================================================================
# 5. Log10-transform metabolites
# =============================================================================
train_log = X_train.copy()
test_log = X_test.copy()

for col in all_met:
    for df in [train_log, test_log]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df[col] = np.log10(df[col])

train_log.to_csv("train_data_log10.csv", index=False)
test_log.to_csv("test_data_log10.csv", index=False)
print("Saved train_data_log10.csv and test_data_log10.csv")
print("\nData preparation complete.")
