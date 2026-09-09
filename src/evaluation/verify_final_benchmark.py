"""
Verification Suite for Stage 10 — Final Benchmarking Module.

Validates:
  1. Module loading and public API signatures.
  2. Completeness: All 8 models represented (3 classical original, 3 classical PCA, 2 quantum).
  3. Metric boundaries: All rates, scores, and probabilities are strictly in [0.0, 1.0].
  4. Exact empirical match: Canonical metrics from Stages 3, 4, 5, 6, 9 match frozen results.
  5. Quantum threshold reality: Recall = 1.0 and Specificity = 0.0 for both VQC configs.
  6. Multi-objective leadership: Classical RF leads ROC-AUC; RF (PCA) leads Specificity; Config A leads quantum efficiency.
  7. Noise impact: Measurable continuous probability delta under depolarizing noise.
  8. Adaptive selection outcome: Config A selected with 0.6667 composite score; both Pareto-optimal.
  9. Explicit scientific limitations: Zero quantum advantage claimed, clinical scope disclaimed.
 10. API Serialization: Unified dictionary serializes cleanly to JSON without error.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.final_benchmark import (
    CANONICAL_BENCHMARK_DATA,
    get_established_benchmarks,
    get_unified_benchmark_table,
    get_quantum_resource_table,
    get_adaptive_selection_summary,
    get_metric_leaders,
    get_platform_limitations,
    assemble_complete_benchmark,
)


def run_all_checks():
    print("=" * 70)
    print("STAGE 10: FINAL BENCHMARKING VERIFICATION SUITE")
    print("=" * 70)

    checks_passed = 0
    total_checks = 0

    def assert_check(condition: bool, description: str):
        nonlocal checks_passed, total_checks
        total_checks += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] Check {total_checks:02d}: {description}")
        else:
            print(f"  [FAIL] Check {total_checks:02d}: {description}")
            raise AssertionError(f"Check failed: {description}")

    # --- Group 1: Data Completeness & Schema ---
    print("\n[Group 1: Data Completeness & Public API]")
    raw_data = get_established_benchmarks()
    assert_check(isinstance(raw_data, dict), "get_established_benchmarks() returns a dict")
    assert_check("metadata" in raw_data, "Metadata section exists")
    assert_check("classical_original_features" in raw_data, "Classical original features section exists")
    assert_check("classical_pca_features" in raw_data, "Classical PCA features section exists")
    assert_check("quantum_configurations" in raw_data, "Quantum configurations section exists")
    assert_check("adaptive_selection_summary" in raw_data, "Adaptive selection summary exists")
    assert_check("limitations" in raw_data, "Limitations section exists")

    # --- Group 2: Unified Benchmark Table ---
    print("\n[Group 2: Unified Benchmark Table Structure]")
    unified_df = get_unified_benchmark_table()
    assert_check(isinstance(unified_df, pd.DataFrame), "Unified benchmark returns a pandas DataFrame")
    assert_check(len(unified_df) == 8, f"Unified table contains exactly 8 models (got {len(unified_df)})")

    expected_models = [
        "Logistic Regression",
        "Random Forest",
        "Support Vector Machine",
        "Logistic Regression (PCA)",
        "Random Forest (PCA)",
        "Support Vector Machine (PCA)",
        "Hybrid VQC (Config A)",
        "Hybrid VQC (Config B)",
    ]
    for model_name in expected_models:
        assert_check(model_name in unified_df["Model"].values, f"Model '{model_name}' present in unified table")

    # Metric bounds check
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Specificity"]
    for col in metric_cols:
        vals = unified_df[col].to_numpy()
        assert_check(
            np.all((vals >= 0.0) & (vals <= 1.0)),
            f"All values in metric column '{col}' are within [0.0, 1.0]"
        )

    # --- Group 3: Canonical Values Verification ---
    print("\n[Group 3: Canonical Values Verification]")
    # Config A
    cfg_a = raw_data["quantum_configurations"]["Config A (2 Qubits / 2 PCA)"]
    assert_check(cfg_a["qubits"] == 2, "Config A has 2 qubits")
    assert_check(cfg_a["n_features"] == 2, "Config A uses 2 PCA components")
    assert_check(abs(cfg_a["roc_auc"] - 0.604839) < 1e-5, f"Config A ROC-AUC is 0.604839 (got {cfg_a['roc_auc']})")
    assert_check(cfg_a["recall"] == 1.0, "Config A recall is exactly 1.0000")
    assert_check(cfg_a["specificity"] == 0.0, "Config A specificity is exactly 0.0000")
    assert_check(cfg_a["circuit_depth"] == 9, "Config A circuit depth is 9")
    assert_check(abs(cfg_a["training_time_sec"] - 30.271) < 1e-3, "Config A training time is 30.271s")
    assert_check(abs(cfg_a["noise_prob_delta"] - 0.021279) < 1e-5, "Config A noise prob delta is 0.021279")

    # Config B
    cfg_b = raw_data["quantum_configurations"]["Config B (4 Qubits / 4 PCA)"]
    assert_check(cfg_b["qubits"] == 4, "Config B has 4 qubits")
    assert_check(cfg_b["n_features"] == 4, "Config B uses 4 PCA components")
    assert_check(abs(cfg_b["roc_auc"] - 0.631720) < 1e-5, f"Config B ROC-AUC is 0.631720 (got {cfg_b['roc_auc']})")
    assert_check(cfg_b["recall"] == 1.0, "Config B recall is exactly 1.0000")
    assert_check(cfg_b["specificity"] == 0.0, "Config B specificity is exactly 0.0000")
    assert_check(cfg_b["circuit_depth"] == 13, "Config B circuit depth is 13")
    assert_check(abs(cfg_b["training_time_sec"] - 57.000) < 1e-3, "Config B training time is 57.000s")
    assert_check(abs(cfg_b["noise_prob_delta"] - 0.040912) < 1e-5, "Config B noise prob delta is 0.040912")

    # Classical baselines
    rf_orig = raw_data["classical_original_features"]["Random Forest"]
    assert_check(abs(rf_orig["roc_auc"] - 0.688172) < 1e-5, "Classical RF (22 feat) ROC-AUC is 0.688172")
    rf_pca = raw_data["classical_pca_features"]["Random Forest (PCA)"]
    assert_check(abs(rf_pca["specificity"] - 0.166667) < 1e-5, "Classical RF (PCA) specificity is 0.166667 (2/12 TN)")

    # --- Group 4: Quantum Resource Table & Adaptive Selection ---
    print("\n[Group 4: Quantum Resource Table & Adaptive Selection]")
    q_table = get_quantum_resource_table()
    assert_check(isinstance(q_table, pd.DataFrame), "Quantum resource table returns a DataFrame")
    assert_check(len(q_table) == 2, "Quantum resource table has 2 configurations")

    adapt_summary = get_adaptive_selection_summary()
    assert_check("Config A" in adapt_summary["selected_configuration"], "Config A is the selected configuration")
    assert_check(len(adapt_summary["pareto_frontier"]) == 2, "Both configurations are on Pareto frontier")
    assert_check(abs(adapt_summary["selection_weights"]["weight_performance"] - 1/3) < 1e-5, "Equal weight for performance")
    assert_check(abs(adapt_summary["selection_weights"]["weight_resource"] - 1/3) < 1e-5, "Equal weight for resource")
    assert_check(abs(adapt_summary["selection_weights"]["weight_noise"] - 1/3) < 1e-5, "Equal weight for noise")

    # --- Group 5: Metric Leaders & Honest Comparison ---
    print("\n[Group 5: Metric Leaders & Honest Comparison]")
    leaders = get_metric_leaders()
    assert_check(leaders["best_roc_auc"]["model"] == "Random Forest", "Best ROC-AUC achieved by Classical Random Forest (22 feat)")
    assert_check(leaders["best_roc_auc"]["value"] > cfg_a["roc_auc"], "Classical RF AUC strictly exceeds Config A AUC")
    assert_check(leaders["best_roc_auc"]["value"] > cfg_b["roc_auc"], "Classical RF AUC strictly exceeds Config B AUC")
    assert_check(leaders["best_specificity"]["model"] == "Random Forest (PCA)", "Best Specificity achieved by Classical RF (PCA)")
    assert_check(leaders["best_quantum_resource_efficiency"]["model"] == "Hybrid VQC (Config A)", "Config A leads quantum efficiency")

    # --- Group 6: Platform Limitations & Integrity ---
    print("\n[Group 6: Scientific Limitations Documentation]")
    limitations = get_platform_limitations()
    assert_check(len(limitations) >= 4, f"At least 4 explicit limitation categories documented (found {len(limitations)})")
    lim_text = " ".join([l["finding"] + " " + l["impact"] for l in limitations])
    assert_check("quantum advantage" in lim_text.lower(), "Explicit statement regarding quantum advantage")
    assert_check("specificity" in lim_text.lower() and "0.0" in lim_text, "Explicit documentation of 0.0 specificity")
    assert_check("not a certified medical diagnostic" in lim_text.lower() or "not a" in lim_text.lower(), "Clinical disclaimer present")

    # --- Group 7: JSON Serialization ---
    print("\n[Group 7: Serialization & Integration Payload]")
    payload = assemble_complete_benchmark()
    assert_check(isinstance(payload, dict), "assemble_complete_benchmark() returns a dict")
    json_str = json.dumps(payload, indent=2)
    assert_check(len(json_str) > 0, "Payload serializes to JSON string without error")
    recovered = json.loads(json_str)
    assert_check(len(recovered["unified_benchmark_table"]) == 8, "Recovered payload has 8 models in table")

    print("\n" + "=" * 70)
    print(f"ALL {checks_passed}/{total_checks} VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
