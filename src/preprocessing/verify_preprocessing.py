"""
Verification script for Stage 2: Parkinson's Dataset & Preprocessing Pipeline.

Executes and verifies dataset loading, validation, subject-aware train/test splitting,
feature scaling, and configurable PCA feature reduction.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.data_loader import load_raw_data, validate_dataset
from src.preprocessing.preprocessor import ParkinsonsPreprocessor


def main():
    print("==================================================")
    print("STAGE 2 VERIFICATION — PARKINSON'S PREPROCESSING")
    print("==================================================\n")

    # 1. Dataset Loading & Validation
    print("--- 1. DATASET ACQUISITION & VALIDATION ---")
    df = load_raw_data()
    val_results = validate_dataset(df)

    print(f"Dataset shape: {df.shape}")
    print(f"Column names ({len(df.columns)}): {list(df.columns)}")
    print(f"Feature count: {val_results['feature_count']}")
    print(f"Target distribution (status): {val_results['target_distribution']}")
    print(f"Missing-value count: {val_results['missing_value_count']}")
    print(f"Number of unique subjects detected: {val_results['unique_subject_count']}\n")

    # 2. Subject-Aware Train/Test Splitting (StandardScaler only)
    print("--- 2. SUBJECT-AWARE SPLITTING & SCALING (WITHOUT PCA) ---")
    preprocessor = ParkinsonsPreprocessor(use_pca=False)
    
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=42
    )

    train_subject_set = set(train_subjects)
    test_subject_set = set(test_subjects)
    subjects_overlap = not train_subject_set.isdisjoint(test_subject_set)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training subject count: {len(train_subject_set)} unique subjects")
    print(f"Test subject count: {len(test_subject_set)} unique subjects")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    print(f"Train/Test subjects overlap? {subjects_overlap}")
    print(f"Confirmation: Train and test subjects are completely disjoint: {not subjects_overlap}\n")

    # Fit and transform features without PCA
    X_train_scaled = preprocessor.fit_transform_train(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    print(f"Scaled training shape: {X_train_scaled.shape}")
    print(f"Scaled test shape: {X_test_scaled.shape}")
    print(f"PCA enabled: {preprocessor.use_pca}\n")

    # 3. Configurable PCA Verification (e.g. n_components=4)
    print("--- 3. CONFIGURABLE PCA VERIFICATION (n_components=4) ---")
    pca_preprocessor = ParkinsonsPreprocessor(use_pca=True, n_components=4, random_state=42)
    X_train_pca, X_test_pca, _, _, _, _ = pca_preprocessor.split_data(df, test_size=0.20, random_state=42)
    
    X_train_pca_trans = pca_preprocessor.fit_transform_train(X_train_pca)
    X_test_pca_trans = pca_preprocessor.transform(X_test_pca)

    pca_info = pca_preprocessor.get_explained_variance_info()
    print(f"PCA enabled: {pca_preprocessor.use_pca}")
    print(f"PCA requested component count: {pca_info['n_components']}")
    print(f"PCA actual training output shape: {X_train_pca_trans.shape}")
    print(f"PCA actual test output shape: {X_test_pca_trans.shape}")
    print("Explained variance ratio per component:")
    for idx, ratio in enumerate(pca_info['explained_variance_ratio'], 1):
        print(f"  Component {idx}: {ratio:.6f} ({ratio*100:.2f}%)")
    print(f"Total explained variance: {pca_info['total_explained_variance']:.6f} ({pca_info['total_explained_variance']*100:.2f}%)\n")

    # 4. Mandatory Validation Rule Checks (Assertions)
    print("--- 4. EXPLICIT MANDATORY VERIFICATION CHECKS ---")
    
    # Rule 1: 'name' is not included in X
    assert "name" not in X_train.columns, "FAILED Rule 1: 'name' identifier column found in X_train!"
    assert "name" not in X_test.columns, "FAILED Rule 1: 'name' identifier column found in X_test!"
    print("✓ Check 1 Passed: 'name' column is excluded from feature matrix X.")

    # Rule 2: 'status' is not included in X
    assert "status" not in X_train.columns, "FAILED Rule 2: 'status' target column found in X_train!"
    assert "status" not in X_test.columns, "FAILED Rule 2: 'status' target column found in X_test!"
    print("✓ Check 2 Passed: 'status' target column is excluded from feature matrix X.")

    # Rule 3: Transformed features are numerical
    assert np.issubdtype(X_train_scaled.dtype, np.number), "FAILED Rule 3: Transformed train features are non-numeric!"
    assert np.issubdtype(X_test_scaled.dtype, np.number), "FAILED Rule 3: Transformed test features are non-numeric!"
    print(f"✓ Check 3 Passed: Transformed features are numerical ({X_train_scaled.dtype}).")

    # Rule 4: Transformed features contain finite values
    assert np.all(np.isfinite(X_train_scaled)), "FAILED Rule 4: Non-finite values (NaN/Inf) found in X_train_scaled!"
    assert np.all(np.isfinite(X_test_scaled)), "FAILED Rule 4: Non-finite values (NaN/Inf) found in X_test_scaled!"
    assert np.all(np.isfinite(X_train_pca_trans)), "FAILED Rule 4: Non-finite values (NaN/Inf) found in X_train_pca!"
    assert np.all(np.isfinite(X_test_pca_trans)), "FAILED Rule 4: Non-finite values (NaN/Inf) found in X_test_pca!"
    print("✓ Check 4 Passed: Transformed features contain only finite values (no NaN or Inf).")

    # Rule 5: No subject appears in both train and test
    assert train_subject_set.isdisjoint(test_subject_set), "FAILED Rule 5: Subject leakage detected between train and test sets!"
    print("✓ Check 5 Passed: No subject appears in both training and testing datasets (isdisjoint verified).")

    # Rule 6: PCA output has requested number of components
    assert X_train_pca_trans.shape[1] == 4, f"FAILED Rule 6: Expected 4 PCA components, got {X_train_pca_trans.shape[1]}"
    assert X_test_pca_trans.shape[1] == 4, f"FAILED Rule 6: Expected 4 PCA components, got {X_test_pca_trans.shape[1]}"
    print("✓ Check 6 Passed: PCA output matches requested component count (4 components).")

    # Rule 7: Median imputation strategy configured
    test_dummy_df = pd.DataFrame({"feat1": [1.0, np.nan, 3.0], "feat2": [2.0, 4.0, 6.0]})
    dummy_prep = ParkinsonsPreprocessor()
    dummy_prep.fit_transform_train(test_dummy_df)
    assert dummy_prep.imputer is not None and dummy_prep.imputer.strategy == "median", "FAILED Rule 7: Imputer strategy is not median!"
    print("✓ Check 7 Passed: Numerical imputation strategy is confirmed as 'median'.")

    print("\n==================================================")
    print("ALL STAGE 2 PREPROCESSING VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
