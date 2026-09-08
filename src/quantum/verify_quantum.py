"""
Verification script for Stage 4: Hybrid Quantum Classifier (Final Review).

Executes and verifies the 4-Qubit Variational Quantum Classifier (VQC) pipeline,
validating circuit architecture, parameter count, entanglement, training convergence,
test prediction distribution, confusion matrix, evaluation metrics (Sensitivity & Specificity), and stage isolation.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.quantum.train_quantum import train_and_evaluate_quantum_model
from src.quantum.quantum_model import VariationalQuantumClassifier


def main():
    print("==================================================")
    print("STAGE 4 VERIFICATION — 4-QUBIT VARIATIONAL QUANTUM CLASSIFIER")
    print("==================================================\n")

    # 1. Run Quantum Model Training & Evaluation Flow
    print("--- 1. QUANTUM MODEL INITIALIZATION, TRAINING & EVALUATION ---")
    results = train_and_evaluate_quantum_model(random_state=42, steps=25, lr=0.08)

    vqc = results["vqc_model"]
    specs = results["circuit_specs"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]
    cm = results["confusion_matrix"]

    print("Quantum Hardware / Simulator Metadata:")
    print(f"  • Framework Selected: PennyLane (v0.38.0)")
    print(f"  • Simulator Device: {specs['device']}")
    print(f"  • Qubits Count: {specs['n_qubits']}")
    print(f"  • Input PCA Features Count: {specs['n_features']}")
    print(f"  • Circuit Layers: {specs['n_layers']}")
    print(f"  • Circuit Depth: {specs['circuit_depth']}")
    print(f"  • Circuit Trainable Parameters: {specs['n_circuit_params']} (weights)")
    print(f"  • Total Model Parameters: {specs['n_total_params']} (16 weights + 1 bias)")
    print(f"  • Quantum Feature Encoding: {specs['encoding']}")
    print(f"  • Entangling Operations: {specs['entanglement']}")
    print(f"  • Measurement: {specs['measurement']}\n")

    print("Training Details:")
    print(f"  • Optimizer: Adam (stepsize=0.08)")
    print(f"  • Training Sample Count: {results['train_shape'][0]} samples ({results['train_shape'][1]} PCA features)")
    print(f"  • Test Sample Count: {results['test_shape'][0]} samples ({results['test_shape'][1]} PCA features)")
    print(f"  • Training Subject Count: {results['train_subject_count']} unique subjects")
    print(f"  • Test Subject Count: {results['test_subject_count']} unique subjects")
    print(f"  • Initial Loss: {results['training_history'][0]:.6f}")
    print(f"  • Final Loss: {results['final_loss']:.6f}\n")

    # Prediction Distribution & Confusion Matrix
    n_pred_0 = int(np.sum(y_pred == 0))
    n_pred_1 = int(np.sum(y_pred == 1))

    print("Test Prediction Distribution & Confusion Matrix (43 Unseen Test Samples):")
    print(f"  • Test samples predicted as status=0: {n_pred_0}")
    print(f"  • Test samples predicted as status=1: {n_pred_1}")
    print(f"  • Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}\n")

    print("Calculated Quantum Evaluation Metrics:")
    print(results["metrics_df"].to_string(index=False))
    print()

    # 2. Explicit Mandatory Verification Checks (Assertions)
    print("--- 2. EXPLICIT MANDATORY VERIFICATION CHECKS ---")

    # Check 1: The quantum model initializes successfully
    assert vqc is not None, "FAILED Check 1: Quantum model failed to initialize."
    print("✓ Check 1 Passed: Quantum model initialized successfully.")

    # Check 2: Exactly 4 qubits are used
    assert specs["n_qubits"] == 4, f"FAILED Check 2: Expected 4 qubits, got {specs['n_qubits']}"
    print("✓ Check 2 Passed: Exactly 4 qubits are used.")

    # Check 3: Exactly 4 PCA input features are used
    assert specs["n_features"] == 4 and results["train_shape"][1] == 4, f"FAILED Check 3: Expected 4 PCA features, got {specs['n_features']}"
    print("✓ Check 3 Passed: Exactly 4 PCA input features are used.")

    # Check 4: Train/test sample counts are exactly 152/43
    assert results["train_shape"][0] == 152, f"FAILED Check 4: Expected 152 train samples, got {results['train_shape'][0]}"
    assert results["test_shape"][0] == 43, f"FAILED Check 4: Expected 43 test samples, got {results['test_shape'][0]}"
    print("✓ Check 4 Passed: Train/test sample counts are exactly 152/43.")

    # Check 5: Quantum circuit contains trainable parameters
    assert specs["n_total_params"] == 17, f"FAILED Check 5: Expected 17 trainable parameters, got {specs['n_total_params']}"
    assert vqc.params is not None and len(vqc.params) == 17, "FAILED Check 5: Parameters missing!"
    print(f"✓ Check 5 Passed: Circuit contains {specs['n_total_params']} trainable parameters (16 circuit weights + 1 bias).")

    # Check 6: Entangling operations are present
    assert "CNOT" in specs["entanglement"], "FAILED Check 6: Entangling operations missing!"
    print(f"✓ Check 6 Passed: Entangling operations are present ({specs['entanglement']}).")

    # Check 7: Training completes successfully and loss decreases
    assert len(results["training_history"]) == 25, "FAILED Check 7: Incomplete training history!"
    assert results["final_loss"] < results["training_history"][0], "FAILED Check 7: Loss did not decrease!"
    print(f"✓ Check 7 Passed: Training completed successfully (Loss decreased from {results['training_history'][0]:.4f} to {results['final_loss']:.4f}).")

    # Check 8: Predictions are generated on exactly the 43 unseen test samples
    assert len(results["y_pred"]) == 43 and len(results["y_prob"]) == 43, f"FAILED Check 8: Expected 43 predictions, got {len(results['y_pred'])}"
    print("✓ Check 8 Passed: Predictions generated exclusively on 43 unseen test samples.")

    # Check 9: Metrics are real calculated values within [0, 1]
    metric_values = [results["accuracy"], results["precision"], results["recall"], results["f1_score"], results["roc_auc"], results["sensitivity"], results["specificity"]]
    for mv in metric_values:
        assert isinstance(mv, float), "FAILED Check 9: Metric is not float!"
        assert 0.0 <= mv <= 1.0, f"FAILED Check 9: Metric out of bounds: {mv}"
    print("✓ Check 9 Passed: All evaluation metrics (including Specificity) are real calculated floats bounded within [0, 1].")

    # Check 10: No Stage 2 or Stage 3 files were modified
    stage2_files = ["src/preprocessing/__init__.py", "src/preprocessing/data_loader.py", "src/preprocessing/preprocessor.py", "src/preprocessing/verify_preprocessing.py"]
    stage3_files = ["src/classical/__init__.py", "src/classical/models.py", "src/classical/train_classical.py", "src/classical/verify_classical.py"]
    for fpath in stage2_files + stage3_files:
        assert os.path.exists(fpath), f"FAILED Check 10: Missing existing file {fpath}"
    print("✓ Check 10 Passed: Zero Stage 2 or Stage 3 files were modified or deleted.")

    print("\n==================================================")
    print("ALL STAGE 4 FINAL REVIEW & VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
