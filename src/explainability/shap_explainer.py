"""
Explainability Layer using SHAP — Stage 8.

Implements model interpretability for the existing Stage 3 classical Random Forest
baseline using TreeExplainer. Explains model behavior across the 22 original
biomedical voice features:
  1. Global Feature Importance: mean absolute SHAP value per feature
  2. Per-Sample Explanations: directional feature contributions (positive vs negative)
  3. Model Consistency / Additivity: base_value + sum(SHAP) == predicted_probability
  4. Precise scientific framing: model explanation, NOT biological causality.

Reuses Stage 2 subject-aware preprocessing pipeline without PCA (original features)
and Stage 3 Random Forest classifier.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.preprocessor import ParkinsonsPreprocessor
from src.classical.models import get_classical_models, evaluate_model


def explain_random_forest(random_state: int = 42) -> Dict[str, Any]:
    """
    Computes global and per-sample SHAP explanations for the Stage 3 Random Forest
    baseline trained on the 22 original biomedical voice measurements:
      1. Reuses Stage 2 subject-aware train/test split.
      2. Reuses StandardScaler fitted strictly on training data (no PCA).
      3. Fits Stage 3 Random Forest on training set (152 rows).
      4. Uses shap.TreeExplainer to compute exact feature attributions on test set (43 rows).
      5. Selects positive class explanations corresponding to status = 1.
      6. Verifies mathematical additivity in probability space.
    """
    # 1. Load dataset
    df = load_raw_data()

    # 2. Stage 2 preprocessing with original 22 features (use_pca=False)
    preprocessor = ParkinsonsPreprocessor(use_pca=False, random_state=random_state)
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=random_state
    )

    # 3. Fit scaling strictly on training data
    X_train_proc = preprocessor.fit_transform_train(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = list(preprocessor.feature_names)

    # 4. Instantiate and train Stage 3 Random Forest
    rf = get_classical_models(random_state=random_state)["Random Forest"]
    eval_res = evaluate_model(rf, X_train_proc, y_train, X_test_proc, y_test)

    # Verify model.classes_ for positive class index
    classes = getattr(rf, "classes_", np.array([0, 1]))
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = 1

    # 5. Compute SHAP values with TreeExplainer
    explainer = shap.TreeExplainer(rf)
    raw_shap_values = explainer.shap_values(X_test_proc)

    # 6. Extract positive class (status = 1) SHAP values and base value
    if isinstance(raw_shap_values, list):
        shap_values_pos = raw_shap_values[pos_idx]
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        shap_values_pos = raw_shap_values[:, :, pos_idx]
    else:
        shap_values_pos = raw_shap_values

    if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value) > 1:
        base_value_pos = float(explainer.expected_value[pos_idx])
    else:
        base_value_pos = float(explainer.expected_value)

    # 7. Global Feature Importance: mean(|SHAP|) across all test samples
    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
    mean_signed_shap = np.mean(shap_values_pos, axis=0)
    std_shap = np.std(shap_values_pos, axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
        "std_shap": std_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df = importance_df[["rank", "feature", "mean_abs_shap", "mean_signed_shap", "std_shap"]]

    # 8. Additivity Check in Probability Space
    # For probability-output TreeExplainer: base_value + sum(shap_values_pos, axis=1) == predict_proba[:, 1]
    reconstructed_proba = base_value_pos + np.sum(shap_values_pos, axis=1)
    actual_proba = eval_res["y_prob"]
    additivity_diffs = np.abs(reconstructed_proba - actual_proba)
    max_additivity_diff = float(np.max(additivity_diffs))
    mean_additivity_diff = float(np.mean(additivity_diffs))
    additivity_passed = bool(max_additivity_diff < 1e-5)

    # 9. Per-Sample Explanations
    sample_explanations = []
    y_true_arr = y_test.values
    y_pred_arr = eval_res["y_pred"]

    for i in range(len(X_test_proc)):
        row_shap = shap_values_pos[i]
        sample_diff = float(additivity_diffs[i])

        # Rank features by contribution
        sorted_indices = np.argsort(row_shap)
        top_negative_indices = sorted_indices[:3]  # most negative (pushing toward status=0)
        top_positive_indices = sorted_indices[::-1][:3]  # most positive (pushing toward status=1)

        top_positive_contributors = [
            {
                "feature": feature_names[idx],
                "shap_value": float(row_shap[idx]),
                "original_feature_value": float(X_test.iloc[i][feature_names[idx]]),
                "scaled_feature_value": float(X_test_proc[i, idx]),
                "interpretation": "Contributed toward the model's status=1 (Parkinson's) output",
            }
            for idx in top_positive_indices if row_shap[idx] > 0
        ]

        top_negative_contributors = [
            {
                "feature": feature_names[idx],
                "shap_value": float(row_shap[idx]),
                "original_feature_value": float(X_test.iloc[i][feature_names[idx]]),
                "scaled_feature_value": float(X_test_proc[i, idx]),
                "interpretation": "Contributed away from status=1 (toward status=0 / Healthy) output",
            }
            for idx in top_negative_indices if row_shap[idx] < 0
        ]

        sample_explanations.append({
            "sample_index": i,
            "actual_class": int(y_true_arr[i]),
            "predicted_class": int(y_pred_arr[i]),
            "predicted_probability": float(actual_proba[i]),
            "reconstructed_probability": float(reconstructed_proba[i]),
            "base_value": base_value_pos,
            "additivity_diff": sample_diff,
            "top_positive_contributors": top_positive_contributors,
            "top_negative_contributors": top_negative_contributors,
            "all_feature_shap_values": {feature_names[j]: float(row_shap[j]) for j in range(len(feature_names))},
        })

    return {
        "model_name": "Random Forest",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_samples": len(X_train_proc),
        "test_samples": len(X_test_proc),
        "train_subjects_count": len(set(train_subjects)),
        "test_subjects_count": len(set(test_subjects)),
        "classes": classes.tolist(),
        "pos_class_index": pos_idx,
        "base_value": base_value_pos,
        "model_evaluation": {
            "accuracy": eval_res["accuracy"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1_score": eval_res["f1_score"],
            "roc_auc": eval_res["roc_auc"],
        },
        "global_importance_df": importance_df,
        "additivity_verification": {
            "output_space": "Probability space [0, 1]",
            "max_absolute_difference": max_additivity_diff,
            "mean_absolute_difference": mean_additivity_diff,
            "tolerance": 1e-5,
            "passed": additivity_passed,
        },
        "sample_explanations": sample_explanations,
        "raw_shap_matrix": shap_values_pos,
        "X_test_processed": X_test_proc,
        "X_test_original": X_test,
        "y_test": y_test,
        "shap_version": shap.__version__,
    }
