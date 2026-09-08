"""
Classical ML model definitions and evaluation helper for Parkinson's disease detection.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_classical_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates and returns the three required classical ML baseline classifiers.
    
    1. Logistic Regression
    2. Random Forest
    3. Support Vector Machine (SVM)
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            random_state=random_state,
            probability=True
        ),
    }


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Trains model on training data and calculates real performance metrics on test data.
    
    Explicitly verifies model.classes_ and extracts the exact continuous probability/score
    corresponding to positive class status=1 for ROC-AUC evaluation.
    """
    # Fit model strictly on training data
    model.fit(X_train, y_train)

    # Predict discrete class labels on unseen test data
    y_pred = model.predict(X_test)
    
    # Inspect model.classes_ to identify index of positive class status=1
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = 1

    # Obtain continuous score/probability for positive class status=1
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        y_prob = probabilities[:, pos_idx]
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_test)
        # If binary classification with classes_=[1, 0], invert decision scores
        if len(classes) == 2 and classes[0] == 1:
            y_prob = -decision_scores
        else:
            y_prob = decision_scores
    else:
        y_prob = y_pred

    # Ensure y_test contains both binary classes
    unique_y_test = set(y_test.unique())
    if not {0, 1}.issubset(unique_y_test):
        raise ValueError(f"y_test does not contain both binary classes {unique_y_test}")

    # Calculate metrics from real predictions
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    sensitivity = rec  # Sensitivity for positive Parkinson's class (status=1)

    return {
        "model": model,
        "classes": classes.tolist(),
        "pos_class_index": pos_idx,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
