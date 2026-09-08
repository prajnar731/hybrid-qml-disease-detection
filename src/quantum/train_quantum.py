"""
Training and evaluation runner for 4-Qubit Variational Quantum Classifier (VQC).
"""

from typing import Dict, Any
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


def train_and_evaluate_quantum_model(
    random_state: int = 42,
    steps: int = 25,
    lr: float = 0.08
) -> Dict[str, Any]:
    """
    Pipeline runner:
    1. Loads raw Parkinson's dataset via Stage 2 loader.
    2. Applies subject-aware split and 4-component PCA preprocessing.
    3. Fits 4-qubit VQC strictly on 152 training samples.
    4. Evaluates VQC model on 43 unseen test samples.
    5. Returns real calculated evaluation metrics (including Specificity) and model metadata.
    """
    # 1. Load raw dataset
    df = load_raw_data()

    # 2. Stage 2 Preprocessing with 4-component PCA
    preprocessor = ParkinsonsPreprocessor(
        use_pca=True,
        n_components=4,
        random_state=random_state
    )

    # 3. Subject-aware train/test split (152 train, 43 test)
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=random_state
    )

    # 4. Preprocessing fitted strictly on training set
    X_train_pca = preprocessor.fit_transform_train(X_train)
    X_test_pca = preprocessor.transform(X_test)

    # 5. Instantiate 4-Qubit VQC Model
    vqc = VariationalQuantumClassifier(
        n_qubits=4,
        n_layers=2,
        random_state=random_state
    )

    # 6. Train model strictly on training data
    history = vqc.fit(X_train_pca, y_train, steps=steps, lr=lr)

    # 7. Predict on unseen test data
    y_prob = vqc.predict_proba(X_test_pca)
    y_pred = vqc.predict(X_test_pca)

    # 8. Calculate real evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    sensitivity = rec

    # Calculate confusion matrix and Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    specs = vqc.get_circuit_specs()

    metrics_df = pd.DataFrame([{
        "Model": "4-Qubit VQC",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }])

    return {
        "vqc_model": vqc,
        "metrics_df": metrics_df,
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
        "final_loss": history[-1] if history else None,
        "training_history": history,
        "train_shape": X_train_pca.shape,
        "test_shape": X_test_pca.shape,
        "train_subject_count": len(set(train_subjects)),
        "test_subject_count": len(set(test_subjects)),
        "circuit_specs": specs,
    }
