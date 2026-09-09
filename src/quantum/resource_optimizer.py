"""
Adaptive Quantum Resource Evaluation — Stage 5.

Experimentally evaluates and compares quantum model configurations across different
qubit and feature allocations on the Parkinson's dataset:
  - Configuration A: 2 PCA components, 2 qubits
  - Configuration B: 4 PCA components, 4 qubits

Directly reuses the frozen Stage 2 subject-aware preprocessing pipeline and the
Stage 4 VariationalQuantumClassifier architecture. Measures actual predictive
performance, circuit depth, parameter count, and training time.
"""

import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.preprocessor import ParkinsonsPreprocessor
from src.quantum.quantum_model import VariationalQuantumClassifier


def evaluate_quantum_configuration(
    n_components: int,
    random_state: int = 42,
    steps: int = 25,
    lr: float = 0.08
) -> Dict[str, Any]:
    """
    Evaluates a single quantum resource configuration end-to-end:
      1. Loads raw Parkinson's dataset.
      2. Performs subject-aware split (same as Stage 2).
      3. Fits PCA with n_components strictly on training data.
      4. Transforms test data using training-fitted PCA.
      5. Instantiates and trains n_components-qubit VQC using Adam optimizer.
      6. Evaluates on unseen 43 test samples.
      7. Records actual measured resources, training time, and evaluation metrics.
    """
    start_total_time = time.time()

    # 1. Load dataset
    df = load_raw_data()

    # 2. Stage 2 Preprocessing with n_components PCA
    preprocessor = ParkinsonsPreprocessor(
        use_pca=True,
        n_components=n_components,
        random_state=random_state
    )

    # 3. Subject-aware train/test split (152 train, 43 test)
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=random_state
    )

    # 4. Fit PCA strictly on training features only
    X_train_pca = preprocessor.fit_transform_train(X_train)
    X_test_pca = preprocessor.transform(X_test)

    # 5. Instantiate VQC with n_qubits = n_components (reusing Stage 4 architecture)
    vqc = VariationalQuantumClassifier(
        n_qubits=n_components,
        n_layers=2,
        random_state=random_state
    )

    # 6. Train VQC strictly on training data and measure wall-clock training time
    start_train_time = time.time()
    loss_history = vqc.fit(X_train_pca, y_train, steps=steps, lr=lr)
    training_time_sec = time.time() - start_train_time

    # 7. Predict on unseen test set
    y_prob = vqc.predict_proba(X_test_pca)
    y_pred = vqc.predict(X_test_pca)

    # 8. Calculate evaluation metrics from real model predictions
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    sensitivity = rec

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 9. Extract actual circuit resources from PennyLane specs
    specs = vqc.get_circuit_specs()
    config_label = f"Configuration {'A' if n_components == 2 else 'B'} ({n_components} Qubits / {n_components} PCA)"

    total_time_sec = time.time() - start_total_time

    return {
        "config_label": config_label,
        "n_pca_features": n_components,
        "n_qubits": specs["n_qubits"],
        "circuit_depth": specs["circuit_depth"],
        "n_circuit_params": specs["n_circuit_params"],
        "n_total_params": specs["n_total_params"],
        "training_time_sec": training_time_sec,
        "total_time_sec": total_time_sec,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_samples": len(X_train_pca),
        "test_samples": len(X_test_pca),
        "train_subjects_count": len(set(train_subjects)),
        "test_subjects_count": len(set(test_subjects)),
        "train_shape": X_train_pca.shape,
        "test_shape": X_test_pca.shape,
        "circuit_specs": specs,
        "vqc_model": vqc,
        "preprocessor": preprocessor,
    }


def run_adaptive_resource_evaluation(
    random_state: int = 42,
    steps: int = 25,
    lr: float = 0.08
) -> Dict[str, Any]:
    """
    Runs the experimental resource evaluation for both required configurations:
      - Configuration A: 2 PCA components, 2 qubits
      - Configuration B: 4 PCA components, 4 qubits
    Returns raw experimental measurements and a side-by-side comparison DataFrame.
    """
    print("==================================================")
    print("RUNNING ADAPTIVE QUANTUM RESOURCE EVALUATION")
    print("==================================================")

    # 1. Run Configuration A (2 PCA features, 2 Qubits)
    print("\n--- Evaluating Configuration A: 2 PCA Components / 2 Qubits ---")
    config_a = evaluate_quantum_configuration(
        n_components=2,
        random_state=random_state,
        steps=steps,
        lr=lr
    )
    print(f"  Configuration A completed in {config_a['training_time_sec']:.2f}s "
          f"(Accuracy: {config_a['accuracy']:.4f}, ROC-AUC: {config_a['roc_auc']:.4f})")

    # 2. Run Configuration B (4 PCA features, 4 Qubits)
    print("\n--- Evaluating Configuration B: 4 PCA Components / 4 Qubits ---")
    config_b = evaluate_quantum_configuration(
        n_components=4,
        random_state=random_state,
        steps=steps,
        lr=lr
    )
    print(f"  Configuration B completed in {config_b['training_time_sec']:.2f}s "
          f"(Accuracy: {config_b['accuracy']:.4f}, ROC-AUC: {config_b['roc_auc']:.4f})")

    # 3. Build side-by-side comparison table
    comparison_rows = [
        {
            "Configuration": "Config A (2 Qubits)",
            "PCA Features": config_a["n_pca_features"],
            "Qubits": config_a["n_qubits"],
            "Circuit Depth": config_a["circuit_depth"],
            "Trainable Params": config_a["n_total_params"],
            "Train Time (s)": round(config_a["training_time_sec"], 3),
            "Accuracy": config_a["accuracy"],
            "Precision": config_a["precision"],
            "Recall": config_a["recall"],
            "F1-Score": config_a["f1_score"],
            "ROC-AUC": config_a["roc_auc"],
            "Sensitivity": config_a["sensitivity"],
            "Specificity": config_a["specificity"],
        },
        {
            "Configuration": "Config B (4 Qubits)",
            "PCA Features": config_b["n_pca_features"],
            "Qubits": config_b["n_qubits"],
            "Circuit Depth": config_b["circuit_depth"],
            "Trainable Params": config_b["n_total_params"],
            "Train Time (s)": round(config_b["training_time_sec"], 3),
            "Accuracy": config_b["accuracy"],
            "Precision": config_b["precision"],
            "Recall": config_b["recall"],
            "F1-Score": config_b["f1_score"],
            "ROC-AUC": config_b["roc_auc"],
            "Sensitivity": config_b["sensitivity"],
            "Specificity": config_b["specificity"],
        }
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "config_a": config_a,
        "config_b": config_b,
        "comparison_df": comparison_df,
    }
