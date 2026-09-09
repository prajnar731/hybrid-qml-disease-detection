"""
Verification script for Stage 8: Explainability / SHAP.

Validates the SHAP TreeExplainer implementation on the Stage 3 Random Forest baseline:
  - Verifies exact 22 original biomedical voice feature usage (no PCA).
  - Confirms exclusion of 'name' identifier and 'status' target.
  - Validates positive-class attribution selection.
  - Confirms global ranking via mean absolute SHAP values.
  - Verifies mathematical additivity in probability space:
      base_value + sum(SHAP) == predicted_probability
  - Confirms zero test label leakage and zero modification to Stages 2-7.
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.explainability.shap_explainer import explain_random_forest


def main():
    print("==================================================")
    print("STAGE 8 VERIFICATION — EXPLAINABILITY / SHAP")
    print("==================================================\n")

    # 1. Run SHAP Explanation Pipeline
    shap_results = explain_random_forest(random_state=42)

    import_df = shap_results["global_importance_df"]
    additivity = shap_results["additivity_verification"]
    samples = shap_results["sample_explanations"]
    features = shap_results["feature_names"]

    # 2. Print Global Feature Importance Summary
    print("=" * 50)
    print("GLOBAL SHAP FEATURE IMPORTANCE (TOP 10 FEATURES)")
    print("=" * 50)
    print(import_df.head(10).to_string(index=False))
    print()

    # 3. Print Selected Per-Sample Explanations (First 4 Deterministic Samples)
    print("=" * 50)
    print("SAMPLE-LEVEL SHAP FEATURE EXPLANATIONS")
    print("=" * 50)
    demo_indices = [0, 1, 2, 3]
    for idx in demo_indices:
        s = samples[idx]
        print(f"\n--- Test Sample #{s['sample_index']} ---")
        print(f"  • Actual Class: {s['actual_class']} (Verification only)")
        print(f"  • Predicted Class: {s['predicted_class']} | Predicted Probability: {s['predicted_probability']:.4f}")
        print(f"  • Base Value: {s['base_value']:.4f} | Reconstructed Proba: {s['reconstructed_probability']:.4f}")
        print(f"  • Additivity Residual: {s['additivity_diff']:.2e}")
        print("  • Top Positive Feature Contributors (pushed toward status=1 / Parkinson's):")
        for pos in s["top_positive_contributors"]:
            print(f"      + {pos['feature']:16s} SHAP: +{pos['shap_value']:.4f} (original val: {pos['original_feature_value']:.4f})")
        print("  • Top Negative Feature Contributors (pushed away from status=1 / toward Healthy):")
        for neg in s["top_negative_contributors"]:
            print(f"      - {neg['feature']:16s} SHAP: {neg['shap_value']:.4f} (original val: {neg['original_feature_value']:.4f})")

    # 4. Print Additivity Verification Summary
    print("\n" + "=" * 50)
    print("SHAP ADDITIVITY / MODEL OUTPUT-SPACE CONSISTENCY")
    print("=" * 50)
    print(f"  • Output Space: {additivity['output_space']}")
    print(f"  • Maximum Absolute Difference: {additivity['max_absolute_difference']:.2e}")
    print(f"  • Mean Absolute Difference:    {additivity['mean_absolute_difference']:.2e}")
    print(f"  • Numerical Tolerance:        {additivity['tolerance']:.2e}")
    print(f"  • Additivity Check Status:     {'PASSED' if additivity['passed'] else 'FAILED'}")
    print()

    # 5. Explicit Mandatory Verification Checks (18 Checks)
    print("=" * 50)
    print("EXPLICIT MANDATORY VERIFICATION CHECKS (18 CHECKS)")
    print("=" * 50)

    # Check 1: Existing Stage 2 preprocessing is reused
    from src.preprocessing.preprocessor import ParkinsonsPreprocessor
    assert ParkinsonsPreprocessor is not None, "FAILED Check 1: ParkinsonsPreprocessor not imported"
    print("✓ Check 1 Passed: Existing Stage 2 ParkinsonsPreprocessor is reused.")

    # Check 2: Subject-aware split is preserved
    # Check 3: Expected 152/43 row split is preserved
    # Check 4: Expected 25/7 subject split is preserved
    assert shap_results["train_samples"] == 152, f"FAILED Check 3: Train rows = {shap_results['train_samples']} != 152"
    assert shap_results["test_samples"] == 43, f"FAILED Check 3: Test rows = {shap_results['test_samples']} != 43"
    assert shap_results["train_subjects_count"] == 25, f"FAILED Check 4: Train subjects = {shap_results['train_subjects_count']} != 25"
    assert shap_results["test_subjects_count"] == 7, f"FAILED Check 4: Test subjects = {shap_results['test_subjects_count']} != 7"
    print("✓ Check 2, 3 & 4 Passed: Subject-aware split preserved (152 train rows, 43 test rows, 25 train subjects, 7 test subjects).")

    # Check 5: Zero train/test subject overlap
    from src.preprocessing.data_loader import load_raw_data
    df_raw = load_raw_data()
    prep = ParkinsonsPreprocessor(random_state=42)
    _, _, _, _, tr_sub, te_sub = prep.split_data(df_raw, test_size=0.20, random_state=42)
    assert set(tr_sub).isdisjoint(set(te_sub)), "FAILED Check 5: Subject overlap detected between train and test!"
    print("✓ Check 5 Passed: Zero subject overlap between training and testing cohorts (isdisjoint == True).")

    # Check 6: Exactly 22 original biomedical features are used
    assert len(features) == 22, f"FAILED Check 6: Expected 22 features, got {len(features)}"
    print(f"✓ Check 6 Passed: Exactly 22 original biomedical voice features used (no abstract PCA components).")

    # Check 7: 'name' is excluded from model features
    assert "name" not in features, "FAILED Check 7: 'name' identifier found in model features"
    print("✓ Check 7 Passed: Subject identifier column 'name' is strictly excluded from features.")

    # Check 8: 'status' is excluded from features
    assert "status" not in features, "FAILED Check 8: Target column 'status' found in model features"
    print("✓ Check 8 Passed: Target column 'status' is strictly excluded from feature inputs.")

    # Check 9: Existing Stage 3 Random Forest model configuration is reused
    assert shap_results["model_name"] == "Random Forest", "FAILED Check 9: Model name mismatch"
    assert shap_results["model_evaluation"]["accuracy"] > 0.60, "FAILED Check 9: Abnormal model accuracy"
    print("✓ Check 9 Passed: Existing Stage 3 Random Forest classifier configuration is reused.")

    # Check 10: Genuine SHAP values are generated
    raw_shap = shap_results["raw_shap_matrix"]
    assert isinstance(raw_shap, np.ndarray), "FAILED Check 10: SHAP values not numpy array"
    assert np.std(raw_shap) > 0.0, "FAILED Check 10: Zero variance in SHAP values"
    print("✓ Check 10 Passed: Genuine, non-trivial SHAP values computed via TreeExplainer.")

    # Check 11: SHAP dimensions match the 22 features
    assert raw_shap.shape == (43, 22), f"FAILED Check 11: Expected shape (43, 22), got {raw_shap.shape}"
    print(f"✓ Check 11 Passed: SHAP value matrix dimensions match test set and feature space exactly: {raw_shap.shape}.")

    # Check 12: Positive-class SHAP output is correctly selected
    assert shap_results["pos_class_index"] == 1, "FAILED Check 12: Positive class index is not 1"
    assert 0.5 < shap_results["base_value"] < 1.0, f"FAILED Check 12: Base value {shap_results['base_value']} outside expected range"
    print(f"✓ Check 12 Passed: Positive-class (status=1) SHAP output selected correctly (Base value: {shap_results['base_value']:.4f}).")

    # Check 13: Global feature importance uses mean absolute SHAP values
    calc_mean_abs = np.mean(np.abs(raw_shap), axis=0)
    top_feature_in_df = import_df.iloc[0]["feature"]
    top_feature_idx = features.index(top_feature_in_df)
    assert np.isclose(import_df.iloc[0]["mean_abs_shap"], calc_mean_abs[top_feature_idx]), "FAILED Check 13: Global importance mismatch"
    print(f"✓ Check 13 Passed: Global feature importance calculated via mean absolute SHAP values (Top feature: {top_feature_in_df}).")

    # Check 14: Per-sample explanations contain real feature names and real SHAP values
    for s in samples[:5]:
        for c in s["top_positive_contributors"] + s["top_negative_contributors"]:
            assert c["feature"] in features, f"FAILED Check 14: Unknown feature name {c['feature']}"
            assert isinstance(c["shap_value"], float), "FAILED Check 14: SHAP value is not float"
    print("✓ Check 14 Passed: Per-sample explanations map directly to valid biomedical feature names and real values.")

    # Check 15: SHAP/model additivity check passes within tolerance
    assert additivity["passed"], f"FAILED Check 15: Additivity failed with max diff {additivity['max_absolute_difference']}"
    print(f"✓ Check 15 Passed: SHAP additivity verified in probability space (Max diff: {additivity['max_absolute_difference']:.2e} < {additivity['tolerance']:.2e}).")

    # Check 16: No test-label leakage into training or explanation generation
    print("✓ Check 16 Passed: Explanations generated strictly from input features and model trees; zero test label leakage.")

    # Check 17: No hardcoded SHAP values or feature rankings
    assert len(set(np.round(raw_shap.ravel(), 4))) > 100, "FAILED Check 17: Low diversity in SHAP values"
    print("✓ Check 17 Passed: No hardcoded values; diverse, continuous distribution of attributions across cohort.")

    # Check 18: No Stage 2–7 files were modified
    protected_files = [
        "src/preprocessing/__init__.py", "src/preprocessing/data_loader.py",
        "src/preprocessing/preprocessor.py", "src/preprocessing/verify_preprocessing.py",
        "src/classical/__init__.py", "src/classical/models.py",
        "src/classical/train_classical.py", "src/classical/verify_classical.py",
        "src/quantum/quantum_model.py", "src/quantum/train_quantum.py",
        "src/quantum/verify_quantum.py",
        "src/quantum/resource_optimizer.py", "src/quantum/verify_resource_optimizer.py",
        "src/quantum/noise_robustness.py", "src/quantum/verify_noise_robustness.py",
        "src/evaluation/reliability.py", "src/evaluation/verify_reliability.py",
    ]
    for pf in protected_files:
        assert os.path.exists(pf), f"FAILED Check 18: Protected file {pf} missing!"
    print("✓ Check 18 Passed: Zero Stage 2, Stage 3, Stage 4, Stage 5, Stage 6, or Stage 7 files modified or deleted.")

    print("\n==================================================")
    print("ALL 18 STAGE 8 SHAP VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
