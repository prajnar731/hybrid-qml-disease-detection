"""
Reliability-Aware Prediction — Stage 7.

Implements multi-signal evidence-based reliability assessment for hybrid quantum
disease detection. Carefully distinguishes between:
  1. Prediction label: (e.g. HIGH RISK / LOW RISK)
  2. Prediction probability / continuous score: (output in [0, 1])
  3. Reliability assessment: (grounded in measurable model behaviors)

Important Scientific Principles:
  - A model output of 0.90 does NOT automatically mean "90% reliable".
  - "Predictive Decisiveness" / "Decision Margin" measures distance from the classification
    boundary (0.5), NOT probability correctness or trustworthy confidence. High decisiveness
    does NOT imply correctness (especially critical given the quantum model's known
    limitation of predicting all 43 test samples as positive with specificity = 0).
  - Brier Score and Expected Calibration Error (ECE) are evaluated as MODEL-LEVEL
    calibration metrics across the evaluation cohort, rather than treated as an individual
    patient's calibration score.
  - Per-sample reliability is evaluated using genuine per-sample signals:
      * Predictive Decisiveness (Decision Margin): S_dec = 2 * |p_q - 0.5|
      * Classical-Quantum Agreement: S_agr = 1 - |p_q - p_c|
      * Noise Stability: S_noise = 1 - |p_ideal - p_noisy|
  - The composite reliability score R is an EXPERIMENTAL RESEARCH METRIC, NOT a
    probability of correctness, NOT clinical confidence, and NOT diagnostic certainty.
  - Reporting categories (HIGH: R >= 0.75, MODERATE: 0.50 <= R < 0.75, LOW: R < 0.50) are
    fixed experimental heuristic thresholds with no clinical meaning.

Zero data leakage: Test labels are never used to train, tune, or optimize models or weights.
All previous stages (2-6) remain frozen and unmodified.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.preprocessor import ParkinsonsPreprocessor
from src.classical.models import get_classical_models, evaluate_model
from src.quantum.noise_robustness import evaluate_configuration_under_noise


# ------------------------------------------------------------------
# Model-Level Calibration Metrics
# ------------------------------------------------------------------

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes cohort-level Brier score: mean squared difference between predicted
    probability and observed binary label. Lower values indicate better probability calibration.
    Range: [0, 1]. This is a MODEL-LEVEL evaluation metric.
    """
    return float(brier_score_loss(y_true, y_prob))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """
    Computes Expected Calibration Error (ECE) across M equal-width bins in [0, 1].
    Evaluates cohort-level alignment between bin confidence and empirical accuracy.
    This is a MODEL-LEVEL evaluation metric.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob > low) & (y_prob <= high)

        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)


# ------------------------------------------------------------------
# Per-Sample Measurable Signals
# ------------------------------------------------------------------

def compute_decision_margin_signal(y_prob: np.ndarray) -> np.ndarray:
    """
    Computes Predictive Decisiveness (Decision Margin) as normalized distance
    from the uninformative 0.5 classification boundary:
      S_dec = 2.0 * |p - 0.5| in [0, 1]

    CRITICAL SCIENTIFIC DISTINCTION:
      This signal measures how far the model's output is from the boundary (0.5).
      It is NOT calibrated "confidence" and does NOT imply probability correctness.
      A high decisiveness score merely indicates the circuit produced an extreme
      expectation value, which can occur even when the prediction is incorrect.
    """
    return 2.0 * np.abs(y_prob - 0.5)


def compute_agreement_signal(
    y_prob_quantum: np.ndarray,
    y_prob_classical: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Measures concordance between quantum prediction and classical baseline:
      - Continuous probability agreement: S_agr = 1.0 - |p_q - p_c| in [0, 1]
      - Discrete class agreement: I(y_pred_q == y_pred_c) in {0, 1}
      - Population class concordance rate

    NOTE: Agreement indicates cross-model consistency, not guaranteed ground-truth correctness.
    """
    prob_agreement = 1.0 - np.abs(y_prob_quantum - y_prob_classical)
    y_pred_q = (y_prob_quantum >= 0.5).astype(int)
    y_pred_c = (y_prob_classical >= 0.5).astype(int)
    class_agreement = (y_pred_q == y_pred_c).astype(int)
    class_agreement_rate = float(np.mean(class_agreement))
    return prob_agreement, class_agreement, class_agreement_rate


def compute_noise_stability_signal(
    y_prob_ideal: np.ndarray,
    y_prob_noisy: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Measures empirical stability of quantum predictions under realistic gate-level
    depolarizing quantum circuit noise (derived from Stage 6):
      delta_p = |p_ideal - p_noisy| in [0, 1]
      S_noise = 1.0 - delta_p in [0, 1]
    """
    delta_p = np.abs(y_prob_ideal - y_prob_noisy)
    stability = 1.0 - delta_p
    mean_stability = float(np.mean(stability))
    return stability, mean_stability


def compute_composite_reliability_score(
    s_dec: np.ndarray,
    s_agr: np.ndarray,
    s_noise: np.ndarray,
    weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
) -> np.ndarray:
    """
    Computes transparent composite per-sample reliability score R in [0, 1]:
      R = w1 * S_dec + w2 * S_agr + w3 * S_noise

    Where:
      - S_dec: Predictive Decisiveness (distance from 0.5 boundary)
      - S_agr: Classical-Quantum Model Agreement
      - S_noise: Physical Quantum Noise Stability (under gate depolarizing noise)

    Weights are fixed a priori to equal weighting (1/3 each, summing to 1.0).
    Higher score indicates greater multi-signal consistency and stability.

    LIMITATIONS:
      - This is an experimental research score.
      - It is NOT a probability of correctness.
      - It is NOT clinical confidence.
      - It is NOT a diagnostic certainty measure.
    """
    w1, w2, w3 = weights
    assert abs((w1 + w2 + w3) - 1.0) < 1e-6, "Weights must sum to 1.0"
    composite = w1 * s_dec + w2 * s_agr + w3 * s_noise
    return np.clip(composite, 0.0, 1.0)


def classify_reliability_category(score: float) -> str:
    """
    Categorizes reliability score using fixed, pre-defined experimental thresholds:
      - HIGH:     R >= 0.75
      - MODERATE: 0.50 <= R < 0.75
      - LOW:      R < 0.50

    DISCLAIMER:
      These thresholds are heuristic experimental reporting boundaries for research
      comparison. They have NO clinical or diagnostic validation.
    """
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MODERATE"
    else:
        return "LOW"


# ------------------------------------------------------------------
# Structured Prediction Formatter
# ------------------------------------------------------------------

def format_prediction_output(
    sample_id: int,
    y_true: int,
    p_quantum: float,
    p_classical: float,
    p_noisy: float,
    composite_score: float,
    s_dec: float,
    s_agr: float,
    s_noise: float,
    config_label: str,
    classical_model_name: str
) -> Dict[str, Any]:
    """
    Produces structured per-sample screening result containing prediction,
    continuous probability, reliability category, and individual component signals.
    """
    pred_class = 1 if p_quantum >= 0.5 else 0
    pred_label = "HIGH RISK (Parkinson's detected)" if pred_class == 1 else "LOW RISK (Healthy)"
    rel_category = classify_reliability_category(composite_score)

    return {
        "sample_id": int(sample_id),
        "prediction": pred_label,
        "predicted_class": int(pred_class),
        "predicted_probability": round(float(p_quantum), 4),
        "ground_truth": int(y_true),
        "is_correct": bool(pred_class == y_true),
        "reliability_assessment": {
            "category": rel_category,
            "composite_score": round(float(composite_score), 4),
            "disclaimer": "Experimental research reliability score; not a clinical guarantee or probability of correctness.",
        },
        "reliability_components": {
            "predictive_decisiveness_margin": round(float(s_dec), 4),
            "classical_quantum_agreement": round(float(s_agr), 4),
            "noise_stability_signal": round(float(s_noise), 4),
        },
        "classical_baseline_reference": {
            "model_name": classical_model_name,
            "predicted_class": int(1 if p_classical >= 0.5 else 0),
            "predicted_probability": round(float(p_classical), 4),
        },
        "quantum_circuit_execution": {
            "configuration": config_label,
            "ideal_probability": round(float(p_quantum), 4),
            "noisy_probability_p005": round(float(p_noisy), 4),
            "noise_shift_delta": round(float(abs(p_quantum - p_noisy)), 4),
        }
    }


# ------------------------------------------------------------------
# Pipeline Runner
# ------------------------------------------------------------------

def evaluate_reliability_pipeline(
    n_components: int = 4,
    p_noise: float = 0.05,
    random_state: int = 42,
    classical_model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Runs complete Stage 7 Reliability-Aware Prediction pipeline for one quantum configuration:
      1. Evaluates quantum model under ideal & noisy circuits (Stage 6).
      2. Trains & evaluates classical baseline (Stage 3 Random Forest on full 22 features).
      3. Computes 3 genuine per-sample reliability signals (Decisiveness, Agreement, Noise Stability).
      4. Calculates model-level calibration metrics (Brier Score and ECE across test cohort).
      5. Computes composite reliability score and experimental categories.
      6. Formats structured sample predictions.
    """
    # 1. Quantum Model Evaluation under Ideal and Noisy circuits (Stage 6)
    q_eval = evaluate_configuration_under_noise(
        n_components=n_components,
        p_noise=p_noise,
        random_state=random_state
    )

    # 2. Classical Baseline Evaluation (Stage 3)
    df = load_raw_data()
    prep_classical = ParkinsonsPreprocessor(use_pca=False, random_state=random_state)
    X_tr_c, X_te_c, y_tr_c, y_te_c, tr_sub, te_sub = prep_classical.split_data(
        df, test_size=0.20, random_state=random_state
    )
    X_tr_scaled = prep_classical.fit_transform_train(X_tr_c)
    X_te_scaled = prep_classical.transform(X_te_c)

    classical_models = get_classical_models(random_state=random_state)
    if classical_model_name not in classical_models:
        raise ValueError(f"Unknown classical model {classical_model_name}. Options: {list(classical_models.keys())}")
    c_model = classical_models[classical_model_name]
    c_eval = evaluate_model(c_model, X_tr_scaled, y_tr_c, X_te_scaled, y_te_c)

    # 3. Align Test Outputs
    y_true = y_te_c.values
    p_q_ideal = q_eval["ideal_metrics"]["y_prob"]
    p_q_noisy = q_eval["noisy_metrics"]["y_prob"]
    p_c = c_eval["y_prob"]

    # 4. Model-Level Calibration Metrics (Evaluated on test cohort; not used as per-patient score)
    brier_q = compute_brier_score(y_true, p_q_ideal)
    brier_c = compute_brier_score(y_true, p_c)
    ece_q = compute_ece(y_true, p_q_ideal, n_bins=5)
    ece_c = compute_ece(y_true, p_c, n_bins=5)

    # 5. Compute Per-Sample Measurable Signals
    s_dec = compute_decision_margin_signal(p_q_ideal)
    s_agr, class_agr, class_agr_rate = compute_agreement_signal(p_q_ideal, p_c)
    s_noise, mean_stability = compute_noise_stability_signal(p_q_ideal, p_q_noisy)

    # 6. Composite Per-Sample Reliability Score (Equal weights: 1/3 each across per-sample signals)
    composite_scores = compute_composite_reliability_score(s_dec, s_agr, s_noise)
    categories = [classify_reliability_category(s) for s in composite_scores]

    # 7. Category Distribution
    cat_counts = {
        "HIGH": int(sum(1 for c in categories if c == "HIGH")),
        "MODERATE": int(sum(1 for c in categories if c == "MODERATE")),
        "LOW": int(sum(1 for c in categories if c == "LOW")),
    }

    # 8. Structured Predictions for all test samples
    config_label = q_eval["config_label"]
    structured_predictions = []
    for i in range(len(y_true)):
        pred_dict = format_prediction_output(
            sample_id=i,
            y_true=int(y_true[i]),
            p_quantum=float(p_q_ideal[i]),
            p_classical=float(p_c[i]),
            p_noisy=float(p_q_noisy[i]),
            composite_score=float(composite_scores[i]),
            s_dec=float(s_dec[i]),
            s_agr=float(s_agr[i]),
            s_noise=float(s_noise[i]),
            config_label=config_label,
            classical_model_name=classical_model_name
        )
        structured_predictions.append(pred_dict)

    return {
        "config_label": config_label,
        "n_qubits": q_eval["n_qubits"],
        "n_pca_features": q_eval["n_pca_features"],
        "classical_model_name": classical_model_name,
        "p_noise": p_noise,
        "model_level_calibration": {
            "quantum_brier_score": brier_q,
            "quantum_ece": ece_q,
            "classical_brier_score": brier_c,
            "classical_ece": ece_c,
            "description": "Cohort-level calibration metrics. Lower is better for both Brier and ECE.",
        },
        "agreement": {
            "mean_probability_agreement": float(np.mean(s_agr)),
            "class_agreement_rate": class_agr_rate,
            "total_concordant_samples": int(np.sum(class_agr)),
            "total_discordant_samples": int(len(class_agr) - np.sum(class_agr)),
        },
        "noise_stability": {
            "mean_stability_signal": mean_stability,
            "mean_probability_delta": float(np.mean(np.abs(p_q_ideal - p_q_noisy))),
            "max_probability_delta": float(np.max(np.abs(p_q_ideal - p_q_noisy))),
        },
        "predictive_decisiveness": {
            "mean_decisiveness_margin": float(np.mean(s_dec)),
            "min_decisiveness_margin": float(np.min(s_dec)),
            "max_decisiveness_margin": float(np.max(s_dec)),
            "note": "Measures distance from 0.5 decision boundary; does NOT represent probability correctness or confidence.",
        },
        "composite_reliability": {
            "formula": "R = (1/3) * S_decisiveness + (1/3) * S_agreement + (1/3) * S_noise_stability",
            "mean_score": float(np.mean(composite_scores)),
            "min_score": float(np.min(composite_scores)),
            "max_score": float(np.max(composite_scores)),
            "std_score": float(np.std(composite_scores)),
            "category_distribution": cat_counts,
            "disclaimer": "Experimental research score; not a clinical confidence score or diagnostic probability.",
        },
        "test_samples_count": len(y_true),
        "structured_predictions": structured_predictions,
        "raw_signals": {
            "p_quantum_ideal": p_q_ideal,
            "p_quantum_noisy": p_q_noisy,
            "p_classical": p_c,
            "y_true": y_true,
            "s_dec": s_dec,
            "s_agr": s_agr,
            "s_noise": s_noise,
            "composite_scores": composite_scores,
            "categories": categories,
        }
    }


def run_full_reliability_analysis(
    p_noise: float = 0.05,
    random_state: int = 42,
    classical_model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Executes reliability evaluation across both quantum configurations:
      - Configuration A: 2 PCA components, 2 qubits
      - Configuration B: 4 PCA components, 4 qubits
    Returns structured results and side-by-side comparison tables.
    """
    print("==================================================")
    print("RUNNING RELIABILITY-AWARE PREDICTION ANALYSIS (STAGE 7)")
    print("==================================================")

    print("\n--- Evaluating Reliability for Config A (2 Qubits / 2 PCA) ---")
    rel_a = evaluate_reliability_pipeline(
        n_components=2,
        p_noise=p_noise,
        random_state=random_state,
        classical_model_name=classical_model_name
    )

    print("\n--- Evaluating Reliability for Config B (4 Qubits / 4 PCA) ---")
    rel_b = evaluate_reliability_pipeline(
        n_components=4,
        p_noise=p_noise,
        random_state=random_state,
        classical_model_name=classical_model_name
    )

    # Build Comparison DataFrame
    rows = []
    for r in [rel_a, rel_b]:
        c = r["composite_reliability"]
        cal = r["model_level_calibration"]
        rows.append({
            "Configuration": r["config_label"],
            "Classical Reference": r["classical_model_name"],
            "Quantum Brier (lower=better)": round(cal["quantum_brier_score"], 4),
            "Quantum ECE (lower=better)": round(cal["quantum_ece"], 4),
            "Mean Decisiveness Margin": round(r["predictive_decisiveness"]["mean_decisiveness_margin"], 4),
            "Mean Model Agreement": round(r["agreement"]["mean_probability_agreement"], 4),
            "Class Concordance Rate": f"{r['agreement']['class_agreement_rate']*100:.1f}%",
            "Mean Noise Stability": round(r["noise_stability"]["mean_stability_signal"], 4),
            "Mean Reliability Score": round(c["mean_score"], 4),
            "Score Range [Min, Max]": f"[{c['min_score']:.3f}, {c['max_score']:.3f}]",
            "High Rel Count": c["category_distribution"]["HIGH"],
            "Mod Rel Count": c["category_distribution"]["MODERATE"],
            "Low Rel Count": c["category_distribution"]["LOW"],
        })
    comparison_df = pd.DataFrame(rows)

    return {
        "rel_a": rel_a,
        "rel_b": rel_b,
        "comparison_df": comparison_df,
    }
