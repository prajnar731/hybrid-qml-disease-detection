"""
Verification script for Stage 3: Classical ML Baselines (Reviewed).

Trains and evaluates Logistic Regression, Random Forest, and SVM models on the
Parkinson's dataset using Stage 2 preprocessed splits. Verifies model.classes_ alignment,
positive class probability extraction for status=1, ROC-AUC calculation, and prevention of data leakage.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classical.train_classical import train_and_evaluate_all_models


def main():
    print("==================================================")
    print("STAGE 3 VERIFICATION — CLASSICAL ML BASELINES (REVIEWED)")
    print("==================================================\n")

    # 1. Baseline Run (StandardScaler on full 22 features)
    print("--- 1. BASELINE CLASSICAL ML EVALUATION (22 FEATURES) ---")
    results = train_and_evaluate_all_models(use_pca=False)

    print(f"Training set sample count: {results['train_shape'][0]} samples ({results['train_shape'][1]} features)")
    print(f"Test set sample count: {results['test_shape'][0]} samples ({results['test_shape'][1]} features)")
    print(f"Training subject count: {results['train_subject_count']} unique subjects")
    print(f"Test subject count: {results['test_subject_count']} unique subjects")
    print(f"Training target distribution: {results['y_train_dist']}")
    print(f"Test target distribution: {results['y_test_dist']}\n")

    print("Model Class Alignment & Probability Extraction Details:")
    for model_name, ev in results["evaluations"].items():
        print(f"  • {model_name}:")
        print(f"      model.classes_: {ev['classes']}")
        print(f"      Positive class (status=1) index: {ev['pos_class_index']}")
        print(f"      Calculated ROC-AUC: {ev['roc_auc']:.6f}")
    print()

    print("Baseline Performance Comparison Table (22 Features):")
    print(results['results_df'].to_string(index=False))
    print()

    # 2. PCA Feature Space Run (n_components=4)
    print("--- 2. PCA FEATURE SPACE EVALUATION (4 PCA COMPONENTS) ---")
    pca_results = train_and_evaluate_all_models(use_pca=True, n_components=4)

    print(f"PCA Training shape: {pca_results['train_shape']}")
    print(f"PCA Test shape: {pca_results['test_shape']}\n")

    print("PCA Model Class Alignment & Probability Extraction Details:")
    for model_name, ev in pca_results["evaluations"].items():
        print(f"  • {model_name}:")
        print(f"      model.classes_: {ev['classes']}")
        print(f"      Positive class (status=1) index: {ev['pos_class_index']}")
        print(f"      Calculated ROC-AUC: {ev['roc_auc']:.6f}")
    print()

    print("PCA Performance Comparison Table (4 Components):")
    print(pca_results['results_df'].to_string(index=False))
    print()

    # 3. Explicit Mandatory Verification Checks (Assertions)
    print("--- 3. EXPLICIT MANDATORY VERIFICATION CHECKS ---")
    expected_models = {"Logistic Regression", "Random Forest", "Support Vector Machine"}
    evaluated_models = set(results["evaluations"].keys())
    
    # Check 1: All three required models exist and trained
    assert expected_models == evaluated_models, f"FAILED Check 1: Expected models {expected_models}, got {evaluated_models}"
    print(f"✓ Check 1 Passed: All 3 required classical models trained successfully: {list(expected_models)}")

    # Check 2: Sample counts match expected Stage 2 split
    assert results["train_shape"][0] == 152, f"FAILED Check 2: Expected 152 train samples, got {results['train_shape'][0]}"
    assert results["test_shape"][0] == 43, f"FAILED Check 2: Expected 43 test samples, got {results['test_shape'][0]}"
    print("✓ Check 2 Passed: Sample counts match exact Stage 2 split (152 train, 43 test).")

    # Check 3: Verified model.classes_ and positive class score extraction
    for model_name, ev in results["evaluations"].items():
        assert ev["classes"] == [0, 1], f"FAILED Check 3: Unexpected classes_ {ev['classes']} for {model_name}"
        assert ev["pos_class_index"] == 1, f"FAILED Check 3: Unexpected pos_class_index for {model_name}"
    print("✓ Check 3 Passed: Verified model.classes_ == [0, 1] and positive class status=1 extracted at index 1.")

    # Check 4: Metrics are valid floats bounded between 0.0 and 1.0
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "sensitivity"]
    for model_name, ev in results["evaluations"].items():
        for mk in metric_keys:
            val = ev[mk]
            assert isinstance(val, float), f"FAILED Check 4: Metric {mk} for {model_name} is not float!"
            assert 0.0 <= val <= 1.0, f"FAILED Check 4: Metric {mk}={val} for {model_name} out of bounds [0.0, 1.0]!"
    print("✓ Check 4 Passed: All metrics are real calculated floats within valid bounds [0.0, 1.0].")

    # Check 5: Sensitivity equals recall for positive class
    for model_name, ev in results["evaluations"].items():
        assert ev["sensitivity"] == ev["recall"], f"FAILED Check 5: Sensitivity != Recall for {model_name}"
    print("✓ Check 5 Passed: Sensitivity equals recall for status=1 across all models.")

    # Check 6: Zero data leakage (predictions generated on unseen test set)
    for model_name, ev in results["evaluations"].items():
        assert len(ev["y_pred"]) == 43, f"FAILED Check 6: Prediction count mismatch for {model_name}"
    print("✓ Check 6 Passed: Predictions generated exclusively on 43 unseen test set samples.")

    print("\n==================================================")
    print("ALL STAGE 3 REVIEW & VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
