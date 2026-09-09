"""
Final Benchmarking Module — Stage 10.

Consolidates all experimentally established benchmarks across the platform:
  1. Classical Baselines (Stage 3):
     - Logistic Regression, Random Forest, SVM on 22 original voice features.
     - Logistic Regression, Random Forest, SVM on 4 PCA components.
  2. Hybrid Quantum Classifiers (Stages 4, 5, 6):
     - Configuration A: 2 PCA components, 2 qubits.
     - Configuration B: 4 PCA components, 4 qubits.
  3. Physical Quantum Noise Robustness (Stage 6):
     - Depolarizing noise p = 0.05 on default.mixed backend.
  4. Adaptive Quantum Resource Selection (Stage 9):
     - Multi-objective equal-weighting and empirical Pareto-efficiency frontier.
  5. Critical Scientific Limitations:
     - Explicit documentation of quantum threshold behavior (specificity = 0.0, recall = 1.0).
     - Decoupling of recall from clinical efficacy.
     - Honest reporting of classical vs. quantum trade-offs (no false advantage claims).

Designed as a modular, serializable library directly consumable by future backend APIs.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Canonical Benchmarking Data Repository
# ------------------------------------------------------------------

CANONICAL_BENCHMARK_DATA: Dict[str, Any] = {
    "metadata": {
        "dataset": "UCI Oxford Parkinson's Disease Detection Dataset (ID: 174)",
        "cohort_size": 195,
        "unique_subjects": 32,
        "split_methodology": "Subject-Aware GroupShuffleSplit (GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42))",
        "train_samples": 152,
        "test_samples": 43,
        "train_subjects": 25,
        "test_subjects": 7,
        "subject_overlap": 0,
        "test_class_distribution": {"status_0_healthy": 12, "status_1_parkinsons": 31},
        "preprocessing": "Median imputation + StandardScaler fitted strictly on training data (no leakage)",
    },
    "classical_original_features": {
        "Logistic Regression": {
            "model_name": "Logistic Regression",
            "feature_representation": "22 Original Features",
            "n_features": 22,
            "qubits": None,
            "accuracy": 0.581395,
            "precision": 0.675676,
            "recall": 0.806452,
            "f1_score": 0.735294,
            "roc_auc": 0.572581,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [6, 25]],
        },
        "Random Forest": {
            "model_name": "Random Forest",
            "feature_representation": "22 Original Features",
            "n_features": 22,
            "qubits": None,
            "accuracy": 0.697674,
            "precision": 0.714286,
            "recall": 0.967742,
            "f1_score": 0.821918,
            "roc_auc": 0.688172,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [1, 30]],
        },
        "Support Vector Machine": {
            "model_name": "Support Vector Machine",
            "feature_representation": "22 Original Features",
            "n_features": 22,
            "qubits": None,
            "accuracy": 0.651163,
            "precision": 0.700000,
            "recall": 0.903226,
            "f1_score": 0.788732,
            "roc_auc": 0.354839,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [3, 28]],
        },
    },
    "classical_pca_features": {
        "Logistic Regression (PCA)": {
            "model_name": "Logistic Regression (PCA)",
            "feature_representation": "4 PCA Components",
            "n_features": 4,
            "qubits": None,
            "accuracy": 0.581395,
            "precision": 0.675676,
            "recall": 0.806452,
            "f1_score": 0.735294,
            "roc_auc": 0.577957,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [6, 25]],
        },
        "Random Forest (PCA)": {
            "model_name": "Random Forest (PCA)",
            "feature_representation": "4 PCA Components",
            "n_features": 4,
            "qubits": None,
            "accuracy": 0.720930,
            "precision": 0.743590,
            "recall": 0.935484,
            "f1_score": 0.828571,
            "roc_auc": 0.338710,
            "specificity": 0.166667,
            "confusion_matrix": [[2, 10], [2, 29]],
        },
        "Support Vector Machine (PCA)": {
            "model_name": "Support Vector Machine (PCA)",
            "feature_representation": "4 PCA Components",
            "n_features": 4,
            "qubits": None,
            "accuracy": 0.627907,
            "precision": 0.692308,
            "recall": 0.870968,
            "f1_score": 0.771429,
            "roc_auc": 0.301075,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [4, 27]],
        },
    },
    "quantum_configurations": {
        "Config A (2 Qubits / 2 PCA)": {
            "model_name": "Hybrid VQC (Config A)",
            "feature_representation": "2 PCA Components",
            "n_features": 2,
            "qubits": 2,
            "accuracy": 0.720930,
            "precision": 0.720930,
            "recall": 1.000000,
            "f1_score": 0.837838,
            "roc_auc": 0.604839,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [0, 31]],
            # Resource metrics
            "trainable_circuit_params": 8,
            "total_params": 9,
            "circuit_depth": 9,
            "training_time_sec": 30.271,
            # Noise metrics (Stage 6)
            "noise_probability": 0.05,
            "noise_channel": "DepolarizingChannel (default.mixed)",
            "noise_prob_delta": 0.021279,
            "mean_expval_shift": 0.011188,
            # Adaptive selection metrics (Stage 9)
            "selection_score": 0.6667,
            "is_selected": True,
            "is_pareto_optimal": True,
        },
        "Config B (4 Qubits / 4 PCA)": {
            "model_name": "Hybrid VQC (Config B)",
            "feature_representation": "4 PCA Components",
            "n_features": 4,
            "qubits": 4,
            "accuracy": 0.720930,
            "precision": 0.720930,
            "recall": 1.000000,
            "f1_score": 0.837838,
            "roc_auc": 0.631720,
            "specificity": 0.000000,
            "confusion_matrix": [[0, 12], [0, 31]],
            # Resource metrics
            "trainable_circuit_params": 16,
            "total_params": 17,
            "circuit_depth": 13,
            "training_time_sec": 57.000,
            # Noise metrics (Stage 6)
            "noise_probability": 0.05,
            "noise_channel": "DepolarizingChannel (default.mixed)",
            "noise_prob_delta": 0.040912,
            "mean_expval_shift": 0.068952,
            # Adaptive selection metrics (Stage 9)
            "selection_score": 0.3333,
            "is_selected": False,
            "is_pareto_optimal": True,
        },
    },
    "adaptive_selection_summary": {
        "selection_weights": {
            "weight_performance": 1.0 / 3.0,
            "weight_resource": 1.0 / 3.0,
            "weight_noise": 1.0 / 3.0,
        },
        "selected_configuration": "Config A (2 Qubits / 2 PCA)",
        "selection_rationale": "Under the predefined equal-weight performance/resource/noise criteria, Config A receives the higher composite score (0.6667 vs 0.3333) and is therefore selected.",
        "pareto_frontier": [
            "Config A (2 Qubits / 2 PCA) - Resource-Efficient / Noise-Resilient operating point",
            "Config B (4 Qubits / 4 PCA) - Performance-Maximizing operating point",
        ],
    },
    "limitations": [
        {
            "category": "Quantum Threshold Behavior",
            "finding": "Both 2-qubit and 4-qubit VQCs predict status=1 for all 43 test samples at default threshold 0.5.",
            "impact": "Recall is fixed at 1.0000 while Specificity is 0.0000. Accuracy matches the test set positive prevalence (72.09%). High recall alone is NOT evidence of clinical efficacy.",
        },
        {
            "category": "Classical vs. Quantum Performance",
            "finding": "Classical Random Forest trained on 22 original features achieves higher discriminative ranking (ROC-AUC 0.6882) than both quantum models (0.6048 and 0.6317).",
            "impact": "No quantum advantage or supremacy is claimed. The hybrid quantum platform serves as an exploratory research prototype for near-term quantum biosensing.",
        },
        {
            "category": "Quantum Decoherence & Noise",
            "finding": "Depolarizing noise (p=0.05) measurably contracts quantum expectation values toward zero (mean probability shifts: 0.0213 for Config A, 0.0409 for Config B).",
            "impact": "The models are NOT universally noise-robust; threshold-based class labels remained invariant only because continuous probabilities stayed above 0.5 under the tested noise regime.",
        },
        {
            "category": "Clinical Scope",
            "finding": "Oxford Parkinson's Voice dataset is a demonstration benchmark for vocal acoustic biomarker processing.",
            "impact": "The platform is an engineering and algorithmic research tool, NOT a certified medical diagnostic system.",
        },
    ]
}


# ------------------------------------------------------------------
# Structured Getter & Export Functions
# ------------------------------------------------------------------

def get_established_benchmarks() -> Dict[str, Any]:
    """Returns the complete canonical benchmark data repository."""
    return CANONICAL_BENCHMARK_DATA


def get_unified_benchmark_table() -> pd.DataFrame:
    """
    Assembles all classical and quantum models into a single structured
    comparison table formatted for easy consumption by UI/APIs.
    """
    rows = []
    data = CANONICAL_BENCHMARK_DATA

    # 1. Classical Original Features
    for m in data["classical_original_features"].values():
        rows.append({
            "Model": m["model_name"],
            "Feature Representation": m["feature_representation"],
            "Qubits": "—",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "Specificity": m["specificity"],
        })

    # 2. Classical PCA Features
    for m in data["classical_pca_features"].values():
        rows.append({
            "Model": m["model_name"],
            "Feature Representation": m["feature_representation"],
            "Qubits": "—",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "Specificity": m["specificity"],
        })

    # 3. Hybrid Quantum Configurations
    for m in data["quantum_configurations"].values():
        rows.append({
            "Model": m["model_name"],
            "Feature Representation": m["feature_representation"],
            "Qubits": str(m["qubits"]),
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "Specificity": m["specificity"],
        })

    return pd.DataFrame(rows)


def get_quantum_resource_table() -> pd.DataFrame:
    """Returns dedicated quantum resource, depth, runtime, and noise comparison table."""
    rows = []
    for cfg in CANONICAL_BENCHMARK_DATA["quantum_configurations"].values():
        rows.append({
            "Configuration": cfg["model_name"],
            "Qubits": cfg["qubits"],
            "PCA Components": cfg["n_features"],
            "Circuit Depth": cfg["circuit_depth"],
            "Trainable Circuit Params": cfg["trainable_circuit_params"],
            "Total Model Params": cfg["total_params"],
            "Training Time (s)": cfg["training_time_sec"],
            "Noise Prob (p)": cfg["noise_probability"],
            "Noise Prob Shift (Δp)": cfg["noise_prob_delta"],
            "Ideal ROC-AUC": cfg["roc_auc"],
            "Adaptive Score": cfg["selection_score"],
            "Selected": "★ YES" if cfg["is_selected"] else "NO",
            "Pareto Status": "Non-Dominated" if cfg["is_pareto_optimal"] else "Dominated",
        })
    return pd.DataFrame(rows)


def get_adaptive_selection_summary() -> Dict[str, Any]:
    """Returns the Stage 9 adaptive selection outcome and weights."""
    return CANONICAL_BENCHMARK_DATA["adaptive_selection_summary"]


def get_metric_leaders() -> Dict[str, Any]:
    """
    Identifies the leading model separately by metric without imposing an
    unsupported 'best overall' declaration.
    """
    table = get_unified_benchmark_table()

    best_auc_idx = table["ROC-AUC"].idxmax()
    best_f1_idx = table["F1-Score"].idxmax()
    best_spec_idx = table["Specificity"].idxmax()
    best_acc_idx = table["Accuracy"].idxmax()

    return {
        "best_roc_auc": {
            "model": table.loc[best_auc_idx, "Model"],
            "feature_representation": table.loc[best_auc_idx, "Feature Representation"],
            "value": float(table.loc[best_auc_idx, "ROC-AUC"]),
            "note": "Measures continuous ranking discriminability across thresholds.",
        },
        "best_f1_score": {
            "model": table.loc[best_f1_idx, "Model"],
            "feature_representation": table.loc[best_f1_idx, "Feature Representation"],
            "value": float(table.loc[best_f1_idx, "F1-Score"]),
            "note": "Harmonic mean of precision and recall (driven high by positive test class prevalence).",
        },
        "best_specificity": {
            "model": table.loc[best_spec_idx, "Model"],
            "feature_representation": table.loc[best_spec_idx, "Feature Representation"],
            "value": float(table.loc[best_spec_idx, "Specificity"]),
            "note": "Ability to correctly identify healthy subjects (true negatives).",
        },
        "best_accuracy": {
            "model": table.loc[best_acc_idx, "Model"],
            "feature_representation": table.loc[best_acc_idx, "Feature Representation"],
            "value": float(table.loc[best_acc_idx, "Accuracy"]),
            "note": "Tied among multiple configurations (72.09% matches the 31/43 positive class baseline).",
        },
        "best_quantum_resource_efficiency": {
            "model": "Hybrid VQC (Config A)",
            "qubits": 2,
            "training_time_sec": 30.271,
            "circuit_depth": 9,
            "note": "Lowest quantum footprint with 48% less decoherence shift than Config B.",
        }
    }


def get_platform_limitations() -> List[Dict[str, str]]:
    """Returns documented scientific limitations and operational boundaries."""
    return CANONICAL_BENCHMARK_DATA["limitations"]


def assemble_complete_benchmark() -> Dict[str, Any]:
    """
    Returns unified JSON-serializable benchmark payload for future API integration.
    """
    unified_df = get_unified_benchmark_table()
    resource_df = get_quantum_resource_table()

    return {
        "metadata": CANONICAL_BENCHMARK_DATA["metadata"],
        "unified_benchmark_table": unified_df.to_dict(orient="records"),
        "quantum_resource_table": resource_df.to_dict(orient="records"),
        "adaptive_selection": get_adaptive_selection_summary(),
        "metric_leaders": get_metric_leaders(),
        "limitations": get_platform_limitations(),
    }
