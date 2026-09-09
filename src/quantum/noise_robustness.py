"""
Noise Robustness Analysis — Stage 6.

Evaluates how the 2-qubit (Config A) and 4-qubit (Config B) Variational Quantum
Classifiers behave under simulated quantum noise (depolarizing noise channel on
PennyLane's default.mixed density-matrix backend).

Assesses performance degradation from ideal to noisy simulation across:
  - Accuracy
  - Precision
  - Recall / Sensitivity
  - F1-Score
  - ROC-AUC
  - Specificity
  - Expectation value contraction (depolarization towards zero)

Noise is introduced strictly at the quantum circuit level (gate-level depolarizing channels).
Zero data leakage; zero modification to Stage 2, 3, 4, or 5 files.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import pennylane as qml
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


def build_noisy_qnode(n_qubits: int, n_layers: int = 2):
    """
    Builds and returns a PennyLane QNode executing on 'default.mixed'
    with depolarizing noise channels applied after rotational and entangling gates.
    """
    dev_mixed = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev_mixed)
    def noisy_circuit(weights, features, p_noise):
        # Data Encoding: AngleEmbedding
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")

        # Variational Layers with Gate-Level Depolarizing Noise
        for layer in range(n_layers):
            # Single-qubit rotations + noise
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
                if p_noise > 0.0:
                    qml.DepolarizingChannel(p_noise, wires=i)

            # Entangling CNOT ring + noise
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
                if p_noise > 0.0:
                    qml.DepolarizingChannel(p_noise, wires=i)

        # Measurement: Pauli-Z expectation value on qubit 0
        return qml.expval(qml.PauliZ(0))

    return noisy_circuit, dev_mixed


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates all evaluation metrics from real model predictions."""
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    sensitivity = rec

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_configuration_under_noise(
    n_components: int,
    p_noise: float = 0.05,
    noise_levels: Optional[List[float]] = None,
    random_state: int = 42,
    steps: int = 25,
    lr: float = 0.08
) -> Dict[str, Any]:
    """
    Evaluates one quantum configuration (2-qubit or 4-qubit) under:
      1. Ideal simulation (default.qubit)
      2. Noisy simulation (default.mixed with depolarizing noise p_noise)
      3. Multi-level noise sweep (optional, e.g. [0.01, 0.05, 0.10])

    Preserves exact trained weights to isolate inference robustness to quantum noise.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.10]

    # 1. Load dataset
    df = load_raw_data()

    # 2. Stage 2 subject-aware split + PCA
    preprocessor = ParkinsonsPreprocessor(
        use_pca=True,
        n_components=n_components,
        random_state=random_state
    )
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=random_state
    )
    X_train_pca = preprocessor.fit_transform_train(X_train)
    X_test_pca = preprocessor.transform(X_test)

    # 3. Train VQC on ideal simulator
    vqc = VariationalQuantumClassifier(
        n_qubits=n_components,
        n_layers=2,
        random_state=random_state
    )
    start_train = time.time()
    loss_history = vqc.fit(X_train_pca, y_train, steps=steps, lr=lr)
    training_time_sec = time.time() - start_train

    # 4. Ideal evaluation on test set
    y_prob_ideal = vqc.predict_proba(X_test_pca)
    ideal_metrics = compute_metrics(y_test.values, y_prob_ideal)

    # Extract trained weights and normalized test features
    n_circuit_params = 2 * n_components * 2
    weights = vqc.params[:n_circuit_params].reshape((2, n_components, 2))
    bias = float(vqc.params[n_circuit_params])
    X_test_norm = vqc.transform_features(X_test_pca)

    # 5. Build circuit-level noisy QNode on default.mixed
    noisy_circuit, dev_mixed = build_noisy_qnode(n_qubits=n_components, n_layers=2)

    # 6. Evaluate on primary benchmark noise level (p_noise)
    start_noisy_eval = time.time()
    raw_expvals_noisy = []
    y_prob_noisy_list = []
    for x in X_test_norm:
        expval = float(noisy_circuit(weights, x, p_noise))
        score = expval + bias
        prob = float(1.0 / (1.0 + np.exp(-score)))
        raw_expvals_noisy.append(expval)
        y_prob_noisy_list.append(prob)

    noisy_eval_time_sec = time.time() - start_noisy_eval
    y_prob_noisy = np.array(y_prob_noisy_list)
    noisy_metrics = compute_metrics(y_test.values, y_prob_noisy)

    # Extract ideal expectation values for physical contraction comparison
    raw_expvals_ideal = []
    for x in X_test_norm:
        expval = float(noisy_circuit(weights, x, 0.0))
        raw_expvals_ideal.append(expval)

    # 7. Compute Performance Degradation (ideal - noisy)
    degradation = {
        "accuracy_deg": ideal_metrics["accuracy"] - noisy_metrics["accuracy"],
        "precision_deg": ideal_metrics["precision"] - noisy_metrics["precision"],
        "recall_deg": ideal_metrics["recall"] - noisy_metrics["recall"],
        "f1_score_deg": ideal_metrics["f1_score"] - noisy_metrics["f1_score"],
        "roc_auc_deg": ideal_metrics["roc_auc"] - noisy_metrics["roc_auc"],
        "specificity_deg": ideal_metrics["specificity"] - noisy_metrics["specificity"],
    }

    # 8. Noise Sweep Evaluation (e.g. 0.01, 0.05, 0.10)
    sweep_results = []
    # Include ideal (p=0.0) as baseline
    sweep_results.append({
        "noise_prob": 0.0,
        "accuracy": ideal_metrics["accuracy"],
        "roc_auc": ideal_metrics["roc_auc"],
        "recall": ideal_metrics["recall"],
        "specificity": ideal_metrics["specificity"],
        "mean_expval": float(np.mean(raw_expvals_ideal)),
        "mean_prob": float(np.mean(y_prob_ideal)),
    })
    for p in noise_levels:
        probs_p = []
        expvals_p = []
        for x in X_test_norm:
            ev = float(noisy_circuit(weights, x, p))
            expvals_p.append(ev)
            pr = float(1.0 / (1.0 + np.exp(-(ev + bias))))
            probs_p.append(pr)
        m_p = compute_metrics(y_test.values, np.array(probs_p))
        sweep_results.append({
            "noise_prob": p,
            "accuracy": m_p["accuracy"],
            "roc_auc": m_p["roc_auc"],
            "recall": m_p["recall"],
            "specificity": m_p["specificity"],
            "mean_expval": float(np.mean(expvals_p)),
            "mean_prob": float(np.mean(probs_p)),
        })

    specs = vqc.get_circuit_specs()
    config_label = f"Config {'A' if n_components == 2 else 'B'} ({n_components} Qubits / {n_components} PCA)"

    return {
        "config_label": config_label,
        "n_pca_features": n_components,
        "n_qubits": n_components,
        "noise_model": "DepolarizingChannel (gate-level after rotations and CNOTs)",
        "p_noise_primary": p_noise,
        "circuit_depth": specs["circuit_depth"],
        "n_circuit_params": specs["n_circuit_params"],
        "n_total_params": specs["n_total_params"],
        "training_time_sec": training_time_sec,
        "noisy_eval_time_sec": noisy_eval_time_sec,
        "train_samples": len(X_train_pca),
        "test_samples": len(X_test_pca),
        "train_subjects_count": len(set(train_subjects)),
        "test_subjects_count": len(set(test_subjects)),
        "loss_history": loss_history,
        "ideal_metrics": ideal_metrics,
        "noisy_metrics": noisy_metrics,
        "degradation": degradation,
        "sweep_results": sweep_results,
        "raw_expvals_ideal_mean": float(np.mean(raw_expvals_ideal)),
        "raw_expvals_noisy_mean": float(np.mean(raw_expvals_noisy)),
        "preprocessor": preprocessor,
        "vqc_model": vqc,
    }


def run_noise_robustness_analysis(
    p_noise: float = 0.05,
    noise_levels: Optional[List[float]] = None,
    random_state: int = 42,
    steps: int = 25,
    lr: float = 0.08
) -> Dict[str, Any]:
    """
    Executes the complete Stage 6 Noise Robustness Analysis for:
      - Configuration A: 2 PCA components, 2 qubits
      - Configuration B: 4 PCA components, 4 qubits
    Returns raw results and formatted comparison DataFrames.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.10]

    print("==================================================")
    print("RUNNING NOISE ROBUSTNESS ANALYSIS (STAGE 6)")
    print("==================================================")

    # 1. Evaluate Configuration A (2 Qubits)
    print("\n--- Evaluating Configuration A: 2 Qubits / 2 PCA Under Noise ---")
    config_a = evaluate_configuration_under_noise(
        n_components=2,
        p_noise=p_noise,
        noise_levels=noise_levels,
        random_state=random_state,
        steps=steps,
        lr=lr
    )
    print(f"  Config A Ideal ROC-AUC: {config_a['ideal_metrics']['roc_auc']:.4f} | "
          f"Noisy (p={p_noise}) ROC-AUC: {config_a['noisy_metrics']['roc_auc']:.4f} | "
          f"Degradation: {config_a['degradation']['roc_auc_deg']:+.4f}")

    # 2. Evaluate Configuration B (4 Qubits)
    print("\n--- Evaluating Configuration B: 4 Qubits / 4 PCA Under Noise ---")
    config_b = evaluate_configuration_under_noise(
        n_components=4,
        p_noise=p_noise,
        noise_levels=noise_levels,
        random_state=random_state,
        steps=steps,
        lr=lr
    )
    print(f"  Config B Ideal ROC-AUC: {config_b['ideal_metrics']['roc_auc']:.4f} | "
          f"Noisy (p={p_noise}) ROC-AUC: {config_b['noisy_metrics']['roc_auc']:.4f} | "
          f"Degradation: {config_b['degradation']['roc_auc_deg']:+.4f}")

    # 3. Build Side-by-Side Comparison DataFrame
    rows = []
    for cfg in [config_a, config_b]:
        id_m = cfg["ideal_metrics"]
        no_m = cfg["noisy_metrics"]
        deg = cfg["degradation"]

        # Ideal row
        rows.append({
            "Configuration": cfg["config_label"],
            "Simulation": "Ideal (p=0.0)",
            "Qubits": cfg["n_qubits"],
            "Depth": cfg["circuit_depth"],
            "Accuracy": id_m["accuracy"],
            "Recall (Sens)": id_m["recall"],
            "Specificity": id_m["specificity"],
            "F1-Score": id_m["f1_score"],
            "ROC-AUC": id_m["roc_auc"],
        })
        # Noisy row
        rows.append({
            "Configuration": cfg["config_label"],
            "Simulation": f"Noisy (p={p_noise})",
            "Qubits": cfg["n_qubits"],
            "Depth": cfg["circuit_depth"],
            "Accuracy": no_m["accuracy"],
            "Recall (Sens)": no_m["recall"],
            "Specificity": no_m["specificity"],
            "F1-Score": no_m["f1_score"],
            "ROC-AUC": no_m["roc_auc"],
        })
        # Degradation row
        rows.append({
            "Configuration": cfg["config_label"],
            "Simulation": "Degradation (Δ)",
            "Qubits": cfg["n_qubits"],
            "Depth": cfg["circuit_depth"],
            "Accuracy": deg["accuracy_deg"],
            "Recall (Sens)": deg["recall_deg"],
            "Specificity": deg["specificity_deg"],
            "F1-Score": deg["f1_score_deg"],
            "ROC-AUC": deg["roc_auc_deg"],
        })

    comparison_df = pd.DataFrame(rows)

    # 4. Build Noise Sweep Summary DataFrame
    sweep_rows = []
    for cfg in [config_a, config_b]:
        for sw in cfg["sweep_results"]:
            sweep_rows.append({
                "Configuration": cfg["config_label"],
                "Noise Prob (p)": sw["noise_prob"],
                "Accuracy": sw["accuracy"],
                "ROC-AUC": sw["roc_auc"],
                "Mean ExpVal": round(sw["mean_expval"], 6),
                "Mean Prob": round(sw["mean_prob"], 6),
            })
    sweep_df = pd.DataFrame(sweep_rows)

    return {
        "config_a": config_a,
        "config_b": config_b,
        "comparison_df": comparison_df,
        "sweep_df": sweep_df,
        "p_noise": p_noise,
        "noise_levels": noise_levels,
    }
